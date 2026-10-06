from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


def select_fbeta_threshold(y_true: np.ndarray, probabilities: np.ndarray, beta: float = 2.0) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    beta_sq = beta**2
    scores = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-12)
    return float(thresholds[int(np.nanargmax(scores))])


def recall_at_precision(y_true: np.ndarray, probabilities: np.ndarray, min_precision: float = 0.80) -> float:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    valid = recall[precision >= min_precision]
    return float(valid.max()) if len(valid) else 0.0


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2.0, zero_division=0)),
        "recall_at_precision_80": recall_at_precision(y_true, probabilities, 0.80),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "threshold": float(threshold),
    }
