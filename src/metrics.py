from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def _safe_auc(labels, probs):
    try:
        return float(roc_auc_score(labels, probs))
    except Exception:
        return 0.0


def _safe_pr_auc(labels, probs):
    try:
        return float(average_precision_score(labels, probs))
    except Exception:
        return 0.0


def compute_metrics(labels, probs, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "auc": _safe_auc(labels, probs),
        "pr_auc": _safe_pr_auc(labels, probs),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


def best_threshold_on_dev(labels, probs) -> tuple[float, dict]:
    best_t = 0.5
    best = compute_metrics(labels, probs, best_t)
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = compute_metrics(labels, probs, float(threshold))
        if metrics["macro_f1"] > best["macro_f1"]:
            best_t, best = float(threshold), metrics
    return best_t, best


accuracy = lambda labels, probs, threshold=0.5: compute_metrics(labels, probs, threshold)["accuracy"]
macro_f1 = lambda labels, probs, threshold=0.5: compute_metrics(labels, probs, threshold)["macro_f1"]
recall = lambda labels, probs, threshold=0.5: compute_metrics(labels, probs, threshold)["recall"]
precision = lambda labels, probs, threshold=0.5: compute_metrics(labels, probs, threshold)["precision"]
auc = lambda labels, probs: compute_metrics(labels, probs, 0.5)["auc"]
pr_auc = lambda labels, probs: compute_metrics(labels, probs, 0.5)["pr_auc"]
