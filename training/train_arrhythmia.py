import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import sys
# 현재 경로를 추가하여 ecg_training 모듈을 찾을 수 있게 함
sys.path.append(os.path.join(os.getcwd(), "kaggle/working/final_delivery/code/ecg_training"))

from ecg_training.datasets import load_arrhythmia_metadata, PTBXLMultilabelDataset
from ecg_training.models import PTBXLClassifier, ArrhythmiaSpecialist
from ecg_training.metrics import evaluate_multilabel
from ecg_training.losses import FocalLoss

def mixup_data(x, y, alpha=0.4):
    """베타 분포를 사용하여 데이터와 레이블을 섞습니다."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="기존 PTB-XL 학습 가중치 경로")
    parser.add_argument("--data_dir", type=str, required=True, help="PTB-XL 데이터셋 경로")
    parser.add_argument("--output_dir", type=str, default="runs/arrhythmia_specialist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mixup", action="store_true", help="Mixup 증강 사용 여부")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 데이터 로드 (부정맥 3종 필터링: AFIB, AFLT, PVC)
    print("📦 부정맥 특화 데이터 로딩 중...")
    # 데이터 경로가 압축 해제된 상태여야 함
    try:
        full_df, class_names = load_arrhythmia_metadata(args.data_dir)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        print("💡 PTB-XL 데이터셋이 지정된 경로에 압축 해제되어 있는지 확인하세요.")
        return
    
    # Fold 1-8: Train, 9: Val
    train_df = full_df[full_df["strat_fold"] <= 8]
    val_df = full_df[full_df["strat_fold"] == 9]
    
    train_ds = PTBXLMultilabelDataset(train_df, args.data_dir, 100, 10, "zscore_per_lead")
    val_ds = PTBXLMultilabelDataset(val_df, args.data_dir, 100, 10, "zscore_per_lead")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # 2. 모델 구성 및 가중치 전이
    print("🏗️ 모델 빌드 및 가중치 전이 중...")
    base_model = PTBXLClassifier(input_leads=12, num_classes=5, embedding_dim=256, 
                                 blocks=[2, 2, 2, 2], base_channels=32, dropout=0.2)
    
    try:
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        base_model.load_state_dict(state_dict)
        print("✅ 기존 백본 가중치 로드 성공")
    except Exception as e:
        print(f"⚠️ 가중치 로드 경고: {e} (무작위 초기값으로 진행할 수 있습니다)")
    
    # 백본을 공유하는 전문가 모델 생성
    model = ArrhythmiaSpecialist(backbone=base_model.backbone, 
                                 num_arrhythmia_classes=len(class_names))
    model.to(device)

    # 3. 손실 함수 및 최적화
    # 부정맥 클래스 불균형을 고려하여 Focal Loss 사용
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. 학습 루프
    best_metric = 0
    print(f"🚀 학습 시작 (Target: {class_names}, Mixup: {args.mixup})")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            signals = batch["signal"].to(device)
            targets = batch["target"].to(device)
            
            if args.mixup:
                signals, targets = mixup_data(signals, targets)
                
            outputs = model(signals)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch["signal"].to(device))
                all_preds.append(torch.sigmoid(outputs).cpu())
                all_targets.append(batch["target"])
        
        val_preds_np = torch.cat(all_preds).numpy()
        val_targets_np = torch.cat(all_targets).numpy()
        
        # 기본 임계값 0.5 설정
        thresholds = np.full(len(class_names), 0.5)
        val_metrics = evaluate_multilabel(val_targets_np, val_preds_np, thresholds, class_names)
        
        current_metric = val_metrics["macro_pr_auc"]
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss/len(train_loader):.4f} - Val PR-AUC: {current_metric:.4f}")
        
        if current_metric > best_metric:
            best_metric = current_metric
            save_path = os.path.join(args.output_dir, "arrhythmia_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_pr_auc": best_metric
            }, save_path)
            print(f"⭐ Best 모델 저장 완료: {save_path}")

    print(f"\n✅ 모든 과정 완료! 최종 PR-AUC: {best_metric:.4f}")

if __name__ == "__main__":
    main()
