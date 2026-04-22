import numpy as np


def f_beta_score(precision, recall, beta):
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denom)


def precision_recall_from_threshold(y_true, y_score, threshold):
    pred = (y_score >= threshold).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall


def tune_thresholds(y_true, y_score, class_names, threshold_config):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    start, stop, step = threshold_config["search_grid"]
    grid = np.arange(start, stop + 1e-9, step)
    default_strategy = threshold_config.get("default_strategy", "f1")
    critical_classes = threshold_config.get("critical_classes", {})

    thresholds = []
    traces = {}

    for class_idx, class_name in enumerate(class_names):
        strategy = critical_classes.get(class_name, default_strategy)
        best_threshold = 0.5
        best_value = -1.0
        trace = []
        target = y_true[:, class_idx]
        score = y_score[:, class_idx]
        for threshold in grid:
            precision, recall = precision_recall_from_threshold(target, score, threshold)
            if strategy == "f2":
                objective = f_beta_score(precision, recall, beta=2.0)
            else:
                objective = f_beta_score(precision, recall, beta=1.0)
            trace.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "objective": float(objective),
                    "strategy": strategy,
                }
            )
            if objective > best_value:
                best_value = objective
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
        traces[class_name] = trace

    return np.asarray(thresholds, dtype=np.float32), traces
