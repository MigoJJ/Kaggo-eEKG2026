import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .archives import extract_archive_if_needed, load_json
from .augmentations import ECGAugmenter
from .datasets import MITBIHBeatDataset, extract_mitbih_beats
from .metrics import evaluate_multiclass, write_json
from .models import MITBIHBeatClassifier


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Pretrain beat encoder on MIT-BIH.")
    parser.add_argument("--config", required=True, help="Path to MIT-BIH JSON config.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda:0.")
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


def resolve_num_workers(requested_num_workers, device):
    if device.type != "cuda":
        return 0
    return max(int(requested_num_workers), 0)


def collate_beats(batch):
    signals = torch.stack([item["signal"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    beat_ids = [item["beat_id"] for item in batch]
    return {"signal": signals, "target": targets, "beat_ids": beat_ids}


def save_history_row(path, row):
    file_exists = path.exists()
    with open(path, "a", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(model, loader, criterion, optimizer, device, grad_clip_norm):
    model.train()
    running_loss = 0.0
    total = 0
    for batch in tqdm(loader, desc="train", leave=False):
        signal = batch["signal"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad()
        logits = model(signal)
        loss = criterion(logits, target)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        running_loss += float(loss.item()) * signal.shape[0]
        total += signal.shape[0]
    return running_loss / max(total, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_targets = []
    all_scores = []
    for batch in tqdm(loader, desc="eval", leave=False):
        signal = batch["signal"].to(device)
        logits = model(signal)
        scores = torch.softmax(logits, dim=1).cpu().numpy()
        all_targets.append(batch["target"].numpy())
        all_scores.append(scores)
    return np.concatenate(all_targets, axis=0), np.concatenate(all_scores, axis=0)


def main():
    args = build_arg_parser().parse_args()
    config = load_json(args.config)
    set_seed(int(config["seed"]))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = make_run_dir(config["output_root"], "mitbih_pretrain")
    write_json(run_dir / "config_snapshot.json", config)

    mitbih_root = extract_archive_if_needed(
        archive_path=config["archives"]["mitbih_zip"],
        extract_root=config["archives"]["extract_root"],
    )

    dataset_cfg = config["dataset"]
    class_names = dataset_cfg["aami_classes"]
    left = int(dataset_cfg["beat_window_left"])
    right = int(dataset_cfg["beat_window_right"])
    max_beats_per_record = dataset_cfg.get("max_beats_per_record")

    train_beats, train_labels, train_ids = extract_mitbih_beats(
        mitbih_root,
        dataset_cfg["train_records"],
        left=left,
        right=right,
        max_beats_per_record=max_beats_per_record,
    )
    val_beats, val_labels, val_ids = extract_mitbih_beats(
        mitbih_root,
        dataset_cfg["val_records"],
        left=left,
        right=right,
        max_beats_per_record=max_beats_per_record,
    )

    augmentation = ECGAugmenter(config["augmentation"]) if config["augmentation"]["enabled"] else None
    train_dataset = MITBIHBeatDataset(train_beats, train_labels, train_ids, class_names, augmentation=augmentation)
    val_dataset = MITBIHBeatDataset(val_beats, val_labels, val_ids, class_names, augmentation=None)

    optimization = config["optimization"]
    num_workers = resolve_num_workers(optimization["num_workers"], device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=optimization["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_beats,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=optimization["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_beats,
    )

    model = MITBIHBeatClassifier(
        input_leads=config["model"]["input_leads"],
        num_classes=len(class_names),
        embedding_dim=config["model"]["embedding_dim"],
        blocks=config["model"]["blocks"],
        base_channels=config["model"]["base_channels"],
        dropout=config["model"]["dropout"],
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization["lr"],
        weight_decay=optimization["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimization["epochs"])

    best_metric = -1.0
    epochs_without_improvement = 0
    history_path = run_dir / "history.csv"

    for epoch in range(1, optimization["epochs"] + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=optimization.get("grad_clip_norm"),
        )
        scheduler.step()
        val_true, val_scores = predict(model, val_loader, device)
        val_metrics = evaluate_multiclass(val_true, val_scores, class_names)
        metric = val_metrics["macro_pr_auc"]

        save_history_row(
            history_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_macro_pr_auc": f"{metric:.6f}",
                "val_macro_roc_auc": f"{val_metrics['macro_roc_auc']:.6f}",
                "val_macro_f1": f"{val_metrics['macro_f1']:.6f}",
                "val_macro_recall": f"{val_metrics['macro_recall']:.6f}",
            },
        )

        if metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.backbone.state_dict(),
                    "best_macro_pr_auc": best_metric,
                },
                run_dir / "best_encoder.pt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= optimization["patience"]:
            break

    write_json(
        run_dir / "report.json",
        {
            "best_val_macro_pr_auc": best_metric,
            "class_names": class_names,
            "dataset_sizes": {"train": len(train_dataset), "val": len(val_dataset)},
        },
    )


if __name__ == "__main__":
    main()
