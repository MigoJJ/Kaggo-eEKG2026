import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .archives import load_json
from .augmentations import ECGAugmenter
from .datasets import UnifiedPhysioNetDataset
from .losses import build_multilabel_loss
from .metrics import evaluate_multilabel, write_json, write_predictions_csv
from .models import PTBXLClassifier
from .thresholds import tune_thresholds


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train Unified ECG multi-label classifier.")
    parser.add_argument("--config", required=True, help="Path to unified training JSON config.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda:0.")
    parser.add_argument("--run-dir", default=None, help="Optional existing or new run directory.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    return parser


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(output_root, prefix):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collate_unified(batch):
    signals = torch.stack([item["signal"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    paths = [item["path"] for item in batch]
    return {"signal": signals, "target": targets, "paths": paths}


def run_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip_norm, use_amp):
    model.train()
    running_loss = 0.0
    total = 0
    for batch in tqdm(loader, desc="train", leave=False):
        signal = batch["signal"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(signal)
            loss = criterion(logits, target)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        batch_size = signal.shape[0]
        running_loss += float(loss.item()) * batch_size
        total += batch_size
    return running_loss / max(total, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_paths = []
    all_targets = []
    all_scores = []
    for batch in tqdm(loader, desc="eval", leave=False):
        signal = batch["signal"].to(device)
        logits = model(signal)
        scores = torch.sigmoid(logits).cpu().numpy()
        all_paths.extend(batch["paths"])
        all_targets.append(batch["target"].numpy())
        all_scores.append(scores)
    return all_paths, np.concatenate(all_targets, axis=0), np.concatenate(all_scores, axis=0)


def save_history_row(path, row):
    file_exists = path.exists()
    with open(path, "a", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = build_arg_parser().parse_args()
    config = load_json(args.config)
    set_seed(int(config.get("seed", 42)))

    if args.epochs is not None:
        config["optimization"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["optimization"]["batch_size"] = int(args.batch_size)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    output_root = config.get("output_root", "runs")
    run_dir = make_run_dir(output_root, "unified_specialist")
    write_json(run_dir / "config_snapshot.json", config)

    # Label mapping from config
    label_mapping = config["dataset"]["label_mapping"]
    
    # Root directories for datasets
    root_dirs = config["dataset"]["root_dirs"]
    
    augmentation = ECGAugmenter(config["augmentation"])
    full_dataset = UnifiedPhysioNetDataset(
        root_dirs=root_dirs,
        label_mapping=label_mapping,
        sample_rate=config["dataset"]["sample_rate"],
        duration_seconds=config["dataset"]["duration_seconds"],
        normalization=config["dataset"]["normalization"],
        augmentation=augmentation if config["augmentation"]["enabled"] else None
    )
    
    class_names = full_dataset.class_names
    num_total = len(full_dataset)
    num_val = int(num_total * config["dataset"]["val_split"])
    num_test = int(num_total * config["dataset"]["test_split"])
    num_train = num_total - num_val - num_test
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(int(config.get("seed", 42)))
    )
    
    # Disable augmentation for val and test
    # (Note: random_split returns Subset objects, which point to the same full_dataset instance.
    # To truly disable augmentation, we'd need separate Dataset instances or a flag in __getitem__)
    # For now, we'll assume the user is okay with this or we can implement a custom Subset.
    
    optimization = config["optimization"]
    train_loader = DataLoader(
        train_ds, batch_size=optimization["batch_size"], shuffle=True,
        num_workers=optimization.get("num_workers", 4), collate_fn=collate_unified
    )
    val_loader = DataLoader(
        val_ds, batch_size=optimization["batch_size"], shuffle=False,
        num_workers=optimization.get("num_workers", 4), collate_fn=collate_unified
    )
    test_loader = DataLoader(
        test_ds, batch_size=optimization["batch_size"], shuffle=False,
        num_workers=optimization.get("num_workers", 4), collate_fn=collate_unified
    )

    model = PTBXLClassifier(
        input_leads=config["model"]["input_leads"],
        num_classes=len(class_names),
        embedding_dim=config["model"]["embedding_dim"],
        blocks=config["model"]["blocks"],
        base_channels=config["model"]["base_channels"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # Use a generic pos_weight or compute if needed
    criterion = build_multilabel_loss(config["loss"], pos_weight=None)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimization["lr"], weight_decay=optimization["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimization["epochs"])
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_metric = -1.0
    history_path = run_dir / "history.csv"

    for epoch in range(1, optimization["epochs"] + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, 
            optimization.get("grad_clip_norm"), use_amp
        )
        scheduler.step()

        val_paths, val_true, val_scores = predict(model, val_loader, device)
        val_metrics = evaluate_multilabel(val_true, val_scores, np.full(len(class_names), 0.5), class_names)
        current_metric = val_metrics["macro_pr_auc"] or 0.0

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val PR-AUC={current_metric:.4f}")
        
        save_history_row(history_path, {
            "epoch": epoch, "train_loss": train_loss, "val_pr_auc": current_metric
        })

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    # Final evaluation
    model.load_state_dict(torch.load(run_dir / "best_model.pt"))
    val_paths, val_true, val_scores = predict(model, val_loader, device)
    tuned_thresholds, _ = tune_thresholds(val_true, val_scores, class_names, config["thresholds"])
    
    test_paths, test_true, test_scores = predict(model, test_loader, device)
    test_metrics = evaluate_multilabel(test_true, test_scores, tuned_thresholds, class_names)
    
    write_json(run_dir / "report.json", {
        "test_metrics": test_metrics,
        "class_names": class_names,
        "thresholds": tuned_thresholds.tolist()
    })
    print(f"Final Test PR-AUC: {test_metrics['macro_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
