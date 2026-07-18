"""1단계(정상 사전선별) 분류기 학습 스크립트.

사용법:
  python train_stage1.py --export-dir <PtbxlPreprocessExporter 출력 디렉터리> \
      --ptbxl-csv <ptbxl_database.csv 경로> --epochs 10 --out ../models/stage1_norm.pt

PTB-XL 공식 strat_fold 관례(1-8 train, 9 val, 10 test)를 따른다.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ecgml.data.ptbxl_dataset import PtbxlNormDataset, load_labels
from ecgml.models.stage1_norm import Stage1NormClassifier
from ecgml.models.stage1_norm_transformer import Stage1NormTransformer


def build_model(model_type: str) -> torch.nn.Module:
    if model_type == "cnn":
        return Stage1NormClassifier()
    if model_type == "transformer":
        return Stage1NormTransformer()
    raise ValueError(f"알 수 없는 model_type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--ptbxl-csv", required=True)
    parser.add_argument("--model-type", choices=["cnn", "transformer"], default="cnn",
                         help="cnn: 로컬 CPU용 소형 모델(기본). transformer: GPU(Kaggle P100 등) 학습용 CNN-Transformer.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/stage1_norm.pt")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    manifest = pd.read_csv(export_dir / "manifest.csv", index_col="ecg_id")
    manifest = manifest[manifest["interpretable"]]

    labels = load_labels(Path(args.ptbxl_csv))
    joined = manifest.join(labels, how="inner")

    train_ids = joined[joined.strat_fold <= 8].index.tolist()
    val_ids = joined[joined.strat_fold == 9].index.tolist()
    test_ids = joined[joined.strat_fold == 10].index.tolist()

    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"train NORM 비율={joined.loc[train_ids, 'is_norm'].mean():.3f}")

    train_ds = PtbxlNormDataset(export_dir, labels, train_ids)
    val_ds = PtbxlNormDataset(export_dir, labels, val_ids)
    test_ds = PtbxlNormDataset(export_dir, labels, test_ids)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, model_type={args.model_type}")
    model = build_model(args.model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)
            print(f"  -> best checkpoint 저장: {out_path}")

    model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    acc = (tp + tn) / max(1, (tp + fp + tn + fn))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    print(f"TEST: acc={acc:.4f} sensitivity(NORM recall)={sensitivity:.4f} specificity={specificity:.4f}")
    print(f"  tp={tp} fp={fp} tn={tn} fn={fn}")


if __name__ == "__main__":
    main()
