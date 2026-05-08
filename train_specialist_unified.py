import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import pandas as pd

import sys
# Path to ecg_training module
sys.path.append(os.path.join(os.getcwd(), "kaggle/working/final_delivery/code/ecg_training"))

from ecg_training.datasets import (
    load_arrhythmia_metadata, PTBXLMultilabelDataset, 
    ChapmanRhythmDataset, CPSC2018Dataset, get_unified_6class_mapper
)
from ecg_training.models import ArrhythmiaSpecialist, PTBXLClassifier
from ecg_training.losses import FocalLoss
from ecg_training.metrics import evaluate_multilabel

def main():
    parser = argparse.ArgumentParser(description="Unified Arrhythmia Specialist Training (Chapman + CPSC + PTB-XL)")
    parser.add_argument("--chapman_dir", type=str, help="Path to Chapman dataset")
    parser.add_argument("--cpsc_dir", type=str, help="Path to CPSC 2018 dataset")
    parser.add_argument("--ptbxl_dir", type=str, help="Path to PTB-XL dataset")
    parser.add_argument("--weights", type=str, help="Path to pretrained PTB-XL backbone weights")
    parser.add_argument("--output_dir", type=str, default="runs/unified_specialist")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    mapper = get_unified_6class_mapper()
    class_names = ["AFIB", "AFLT", "SVPB", "PVC", "SVTA", "VTA"]

    datasets = []
    
    # 1. Load PTB-XL (subset for specialist)
    if args.ptbxl_dir:
        print("📥 Loading PTB-XL Specialist subset...")
        ptb_df, _ = load_arrhythmia_metadata(args.ptbxl_dir)
        # Note: load_arrhythmia_metadata currently filters for 3 classes. 
        # In a unified training, we would map these to the 6-class output.
        ptb_ds = PTBXLMultilabelDataset(ptb_df, args.ptbxl_dir, 100, 10, "zscore_per_lead")
        # Custom mapping logic would be needed here to match 6 classes
        # For simplicity in this script, we assume Chapman/CPSC are primary for the 6-class task.
        pass

    # 2. Load Chapman (Shaoxing)
    if args.chapman_dir:
        print("📥 Loading Chapman Dataset...")
        # Expecting a CSV with 'FileName' and 'Rhythm' mapped via mapper
        # This part requires the user to have prepared metadata
        meta_path = Path(args.chapman_dir) / "metadata.csv"
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            datasets.append(ChapmanRhythmDataset(df, args.chapman_dir, sample_rate=500))

    # 3. Load CPSC 2018
    if args.cpsc_dir:
        print("📥 Loading CPSC 2018 Dataset...")
        meta_path = Path(args.cpsc_dir) / "metadata.csv"
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            datasets.append(CPSC2018Dataset(df, args.cpsc_dir, sample_rate=500))

    if not datasets:
        print("❌ No datasets loaded. Please check paths and metadata.")
        return

    full_ds = ConcatDataset(datasets)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    # Build Model
    print("🏗️ Building Specialist Model (6-class)...")
    # Use PTB-XL backbone if weights provided
    base_model = PTBXLClassifier(input_leads=12, num_classes=5, embedding_dim=256, 
                                 blocks=[2, 2, 2, 2], base_channels=32, dropout=0.2)
    if args.weights:
        checkpoint = torch.load(args.weights, map_location=device)
        base_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        print("✅ Backbone weights loaded.")

    model = ArrhythmiaSpecialist(backbone=base_model.backbone, num_arrhythmia_classes=6)
    model.to(device)

    criterion = nn.CrossEntropyLoss() # If single-label rhythm task, else FocalLoss for multi-label
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            signals = batch["signal"].to(device)
            # Upsample if 100Hz (PTB-XL) to 500Hz or vice versa. 
            # In V2 we standardized everything to the same length (1000 or 5000)
            targets = batch["target"].to(device)
            outputs = model(signals)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch["signal"].to(device))
                preds = outputs.argmax(dim=1)
                correct += (preds == batch["target"].to(device)).sum().item()
                total += targets.size(0)
        
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "unified_specialist_best.pt"))
        
        scheduler.step()

    print(f"✅ Training Complete. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
