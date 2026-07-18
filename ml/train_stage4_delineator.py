"""Step 2: train a P/QRS/T delineator from LUDB + QTDB on Kaggle.

Examples:
  python train_stage4_delineator.py \
    --ludb-root /kaggle/input/ludb/ludb/1.0.1 \
    --qtdb-root /kaggle/input/qt-database/qtdb/1.0.0 \
    --epochs 30 --out /kaggle/working/stage4_delineator.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from ecgml.data.wfdb_delineation_dataset import CLASS_NAMES, WfdbDelineationDataset
from ecgml.models.stage4_delineator import Stage4Delineator
from ecgml.torch_device import resolve_device


def pixel_f1(confusion: torch.Tensor, cls: int) -> float:
    tp = confusion[cls, cls].item()
    fp = confusion[:, cls].sum().item() - tp
    fn = confusion[cls, :].sum().item() - tp
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def evaluate(model, loader, device) -> tuple[float, dict[str, float]]:
    model.eval()
    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu()
            for t, p in zip(y.reshape(-1), pred.reshape(-1)):
                confusion[t.long(), p.long()] += 1
    f1 = {name: pixel_f1(confusion, i) for i, name in enumerate(CLASS_NAMES)}
    macro = sum(f1[name] for name in CLASS_NAMES[1:]) / 3
    return macro, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ludb-root")
    parser.add_argument("--qtdb-root")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", default="models/stage4_delineator.pt")
    args = parser.parse_args()

    datasets = []
    if args.ludb_root:
        print("LUDB loading...")
        ludb = WfdbDelineationDataset(args.ludb_root)
        print(f"LUDB usable records={len(ludb)}")
        datasets.append(ludb)
    if args.qtdb_root:
        print("QTDB loading...")
        qtdb = WfdbDelineationDataset(args.qtdb_root)
        print(f"QTDB usable records={len(qtdb)}")
        datasets.append(qtdb)
    if not datasets:
        raise SystemExit("At least one of --ludb-root or --qtdb-root is required.")

    ds = ConcatDataset(datasets)
    n_val = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(2026))
    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    print(f"device={device}")
    model = Stage4Delineator(n_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = torch.tensor([0.05, 1.0, 0.8, 1.0], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_macro = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)
        macro, f1 = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_macro_wave_f1={macro:.4f} f1={f1}")
        if macro > best_macro:
            best_macro = macro
            torch.save(model.state_dict(), out)
            (out.with_suffix(".json")).write_text(json.dumps({
                "version": "stage4_delineator_kaggle_v1",
                "classes": CLASS_NAMES,
                "target_fs": 500,
                "window_samples": 5000,
                "best_val_macro_wave_f1": best_macro,
            }, indent=2))
            print(f"  -> saved {out}")


if __name__ == "__main__":
    main()
