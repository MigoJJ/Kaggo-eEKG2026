import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_binary_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_binary_ap(y_true, y_score):
    if np.sum(y_true) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def specificity_score(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    denom = true_negative + false_positive
    return float(true_negative / denom) if denom > 0 else 0.0


def binary_confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate_multilabel(y_true, y_score, thresholds, class_names):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    y_pred = (y_score >= thresholds[None, :]).astype(np.int32)

    per_class = {}
    auc_values = []
    ap_values = []

    for class_idx, class_name in enumerate(class_names):
        target = y_true[:, class_idx]
        pred = y_pred[:, class_idx]
        score = y_score[:, class_idx]
        auc = safe_binary_auc(target, score)
        ap = safe_binary_ap(target, score)
        recall = float(np.sum((target == 1) & (pred == 1)) / max(float(np.sum(target == 1)), 1.0))
        precision = float(np.sum((target == 1) & (pred == 1)) / max(float(np.sum(pred == 1)), 1.0))
        f1 = float(f1_score(target, pred, zero_division=0))
        bal_acc = float(balanced_accuracy_score(target, pred))
        spec = specificity_score(target, pred)
        conf = binary_confusion(target, pred)
        if auc is not None:
            auc_values.append(auc)
        if ap is not None:
            ap_values.append(ap)
        per_class[class_name] = {
            "roc_auc": auc,
            "pr_auc": ap,
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "specificity": spec,
            "balanced_accuracy": bal_acc,
            "threshold": float(thresholds[class_idx]),
            "confusion": conf,
        }

    macro_f1 = float(np.mean([per_class[name]["f1"] for name in class_names]))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_recall = float(np.mean([per_class[name]["recall"] for name in class_names]))
    macro_specificity = float(np.mean([per_class[name]["specificity"] for name in class_names]))
    macro_balanced_accuracy = float(np.mean([per_class[name]["balanced_accuracy"] for name in class_names]))

    return {
        "macro_roc_auc": float(np.mean(auc_values)) if auc_values else None,
        "macro_pr_auc": float(np.mean(ap_values)) if ap_values else None,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_recall": macro_recall,
        "macro_specificity": macro_specificity,
        "macro_balanced_accuracy": macro_balanced_accuracy,
        "per_class": per_class,
    }


def evaluate_multiclass(y_true, y_score, class_names):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)
    y_pred = np.argmax(y_score, axis=1)
    per_class = {}
    for idx, class_name in enumerate(class_names):
        target = (y_true == idx).astype(np.int32)
        pred = (y_pred == idx).astype(np.int32)
        score = y_score[:, idx]
        per_class[class_name] = {
            "roc_auc": safe_binary_auc(target, score),
            "pr_auc": safe_binary_ap(target, score),
            "f1": float(f1_score(target, pred, zero_division=0)),
            "recall": float(np.sum((target == 1) & (pred == 1)) / max(float(np.sum(target == 1)), 1.0)),
            "precision": float(np.sum((target == 1) & (pred == 1)) / max(float(np.sum(pred == 1)), 1.0)),
        }
    return {
        "macro_roc_auc": float(np.mean([v["roc_auc"] for v in per_class.values() if v["roc_auc"] is not None])),
        "macro_pr_auc": float(np.mean([v["pr_auc"] for v in per_class.values() if v["pr_auc"] is not None])),
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "macro_recall": float(np.mean([v["recall"] for v in per_class.values()])),
        "per_class": per_class,
    }


def write_predictions_csv(path, ids, class_names, y_true, y_score):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="ascii", newline="") as handle:
        fieldnames = ["record_id"]
        for name in class_names:
            fieldnames.extend([f"target__{name}", f"score__{name}"])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record_id in enumerate(ids):
            row = {"record_id": record_id}
            for class_idx, class_name in enumerate(class_names):
                row[f"target__{class_name}"] = int(y_true[idx][class_idx])
                row[f"score__{class_name}"] = f"{float(y_score[idx][class_idx]):.6f}"
            writer.writerow(row)


def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2)
