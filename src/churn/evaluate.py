"""Metrics used to compare every model in the project.

Accuracy is reported but never used to pick a model: predicting that nobody
churns already scores ~73% on this data.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def lift_at(y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.2) -> float:
    """Share of all churners found by contacting the top `top_frac` riskiest customers."""
    y_true = np.asarray(y_true)
    n_top = int(np.ceil(len(y_true) * top_frac))
    top_idx = np.argsort(-np.asarray(y_prob))[:n_top]
    return y_true[top_idx].sum() / y_true.sum()


def classification_report(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "churners_caught_top20pct": lift_at(y_true, y_prob, 0.2),
    }
