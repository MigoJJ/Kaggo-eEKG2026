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
from .datasets import PTBXLMultilabelDataset, create_weighted_sampler, load_ptbxl_metadata
from .losses import build_multilabel_loss, compute_pos_weight
from .metrics import evaluate_multilabel, write_json, write_predictions_csv
from .models import PTBXLClassifier
from .thresholds import tune_thresholds


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train PTB-XL 5-superclass multi-label classifier.")
    parser.add_argument("--config", required=True, help="Path to baseline JSON config.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda:0.")
    parser.add_argument("--run-dir", default=None, help="Optional existing or new run directory.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA.")
    parser.add_argument("--save-every-epoch", action="store_true", help="Save epoch checkpoints as well as latest checkpoint.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional train subset limit.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional validation subset limit.")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional test subset limit.")
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


def resolve_run_dir(output_root, prefix, run_dir_arg, resume_path):
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    if resume_path:
        return Path(resume_path).resolve().parent
    return make_run_dir(output_root, prefix)


def resolve_num_workers(requested_num_workers, device):
    if device.type != "cuda":
        return 0
    return max(int(requested_num_workers), 0)


def collate_ids(batch):
    signals = torch.stack([item["signal"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    ids = [item["ecg_id"] for item in batch]
    return {"signal": signals, "target": targets, "ids": ids}


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
    all_ids = []
    all_targets = []
    all_scores = []
    for batch in tqdm(loader, desc="eval", leave=False):
        signal = batch["signal"].to(device)
        logits = model(signal)
        scores = torch.sigmoid(logits).cpu().numpy()
        all_ids.extend(batch["ids"])
        all_targets.append(batch["target"].numpy())
        all_scores.append(scores)
    return all_ids, np.concatenate(all_targets, axis=0), np.concatenate(all_scores, axis=0)


def save_history_row(path, row):
    file_exists = path.exists()
    with open(path, "a", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_pretrained_encoder_if_available(model, encoder_path):
    if not encoder_path:
        return
    checkpoint = torch.load(encoder_path, map_location="cpu")
    state_dict = checkpoint["encoder_state_dict"] if "encoder_state_dict" in checkpoint else checkpoint
    model.backbone.load_state_dict(state_dict, strict=False)


def maybe_limit_frame(frame, limit):
    if limit is None:
        return frame
    limit = max(int(limit), 1)
    return frame.head(limit).copy()


def resolve_ptbxl_root(archives_config):
    ptbxl_dir = archives_config.get("ptbxl_dir")
    if ptbxl_dir:
        return Path(ptbxl_dir)
    return extract_archive_if_needed(
        archive_path=archives_config["ptbxl_zip"],
        extract_root=archives_config["extract_root"],
    )


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_metric, epochs_without_improvement, config, args):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_macro_pr_auc": best_metric,
        "epochs_without_improvement": epochs_without_improvement,
        "config": config,
        "cli_args": vars(args),
    }


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_resume_checkpoint(resume_path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def main():
    args = build_arg_parser().parse_args()
    config = load_json(args.config)
    set_seed(int(config["seed"]))

    if args.epochs is not None:
        config["optimization"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["optimization"]["batch_size"] = int(args.batch_size)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = resolve_run_dir(config["output_root"], "ptbxl", args.run_dir, args.resume)
    write_json(run_dir / "config_snapshot.json", config)

    ptbxl_root = resolve_ptbxl_root(config["archives"])

    class_names = config["dataset"]["superclasses"]
    metadata = load_ptbxl_metadata(ptbxl_root, class_names)

    train_frame = metadata[metadata["strat_fold"].isin(config["dataset"]["train_folds"])].copy()
    val_frame = metadata[metadata["strat_fold"].isin(config["dataset"]["val_folds"])].copy()
    test_frame = metadata[metadata["strat_fold"].isin(config["dataset"]["test_folds"])].copy()
    train_frame = maybe_limit_frame(train_frame, args.limit_train)
    val_frame = maybe_limit_frame(val_frame, args.limit_val)
    test_frame = maybe_limit_frame(test_frame, args.limit_test)

    augmentation = ECGAugmenter(config["augmentation"])
    train_dataset = PTBXLMultilabelDataset(
        frame=train_frame,
        dataset_dir=ptbxl_root,
        sample_rate=config["dataset"]["sample_rate"],
        duration_seconds=config["dataset"]["duration_seconds"],
        normalization=config["dataset"]["normalization"],
        augmentation=augmentation if config["augmentation"]["enabled"] else None,
    )
    eval_dataset_kwargs = dict(
        dataset_dir=ptbxl_root,
        sample_rate=config["dataset"]["sample_rate"],
        duration_seconds=config["dataset"]["duration_seconds"],
        normalization=config["dataset"]["normalization"],
        augmentation=None,
    )
    val_dataset = PTBXLMultilabelDataset(frame=val_frame, **eval_dataset_kwargs)
    test_dataset = PTBXLMultilabelDataset(frame=test_frame, **eval_dataset_kwargs)

    sampler = None
    shuffle = True
    if config["dataset"].get("use_weighted_sampler", False):
        sampler = create_weighted_sampler(train_frame, class_names)
        shuffle = False

    optimization = config["optimization"]
    num_workers = resolve_num_workers(optimization["num_workers"], device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=optimization["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_ids,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=optimization["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_ids,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=optimization["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_ids,
    )

    model = PTBXLClassifier(
        input_leads=config["model"]["input_leads"],
        num_classes=len(class_names),
        embedding_dim=config["model"]["embedding_dim"],
        blocks=config["model"]["blocks"],
        base_channels=config["model"]["base_channels"],
        dropout=config["model"]["dropout"],
    ).to(device)
    load_pretrained_encoder_if_available(model, config["model"].get("pretrained_encoder_path"))

    pos_weight = compute_pos_weight(train_frame).to(device)
    criterion = build_multilabel_loss(config["loss"], pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization["lr"],
        weight_decay=optimization["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimization["epochs"])
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_metric = -1.0
    best_state = None
    epochs_without_improvement = 0
    start_epoch = 1
    history_path = run_dir / "history.csv"
    latest_checkpoint_path = run_dir / "latest_checkpoint.pt"

    if args.resume:
        resumed = load_resume_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        start_epoch = int(resumed.get("epoch", 0)) + 1
        best_metric = float(resumed.get("best_macro_pr_auc", -1.0))
        epochs_without_improvement = int(resumed.get("epochs_without_improvement", 0))
        best_state = {"model_state_dict": model.state_dict(), "epoch": start_epoch - 1, "best_macro_pr_auc": best_metric}

    for epoch in range(start_epoch, optimization["epochs"] + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip_norm=optimization.get("grad_clip_norm"),
            use_amp=use_amp,
        )
        scheduler.step()

        val_ids, val_true, val_scores = predict(model, val_loader, device)
        default_thresholds = np.full(len(class_names), 0.5, dtype=np.float32)
        val_metrics = evaluate_multilabel(val_true, val_scores, default_thresholds, class_names)
        current_metric = val_metrics["macro_pr_auc"] if val_metrics["macro_pr_auc"] is not None else -1.0

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_macro_pr_auc": "" if val_metrics["macro_pr_auc"] is None else f"{val_metrics['macro_pr_auc']:.6f}",
            "val_macro_roc_auc": "" if val_metrics["macro_roc_auc"] is None else f"{val_metrics['macro_roc_auc']:.6f}",
            "val_macro_f1": f"{val_metrics['macro_f1']:.6f}",
            "val_macro_recall": f"{val_metrics['macro_recall']:.6f}",
            "learning_rate": f"{scheduler.get_last_lr()[0]:.8f}",
        }
        save_history_row(history_path, row)

        if current_metric > best_metric:
            best_metric = current_metric
            best_state = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                config=config,
                args=args,
            )
            torch.save(best_state, run_dir / "best_model.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        latest_state = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            config=config,
            args=args,
        )
        save_checkpoint(latest_checkpoint_path, latest_state)
        if args.save_every_epoch:
            save_checkpoint(run_dir / f"checkpoint_epoch_{epoch:03d}.pt", latest_state)

        if epochs_without_improvement >= optimization["patience"]:
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device)["model_state_dict"])

    val_ids, val_true, val_scores = predict(model, val_loader, device)
    tuned_thresholds, threshold_traces = tune_thresholds(val_true, val_scores, class_names, config["thresholds"])
    val_metrics = evaluate_multilabel(val_true, val_scores, tuned_thresholds, class_names)
    write_predictions_csv(run_dir / "val_predictions.csv", val_ids, class_names, val_true, val_scores)

    test_ids, test_true, test_scores = predict(model, test_loader, device)
    test_metrics = evaluate_multilabel(test_true, test_scores, tuned_thresholds, class_names)
    write_predictions_csv(run_dir / "test_predictions.csv", test_ids, class_names, test_true, test_scores)

    write_json(
        run_dir / "thresholds.json",
        {
            "class_names": class_names,
            "thresholds": {class_name: float(tuned_thresholds[idx]) for idx, class_name in enumerate(class_names)},
            "search_traces": threshold_traces,
        },
    )
    write_json(
        run_dir / "report.json",
        {
            "selection_metric": "macro_pr_auc",
            "best_val_macro_pr_auc_at_threshold_0_5": best_metric,
            "resumed_from": args.resume,
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "val_metrics_tuned": val_metrics,
            "test_metrics_tuned": test_metrics,
            "class_names": class_names,
            "dataset_sizes": {
                "train": len(train_dataset),
                "val": len(val_dataset),
                "test": len(test_dataset),
            },
        },
    )


if __name__ == "__main__":
    main()
