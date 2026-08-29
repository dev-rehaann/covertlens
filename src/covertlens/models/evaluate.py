"""Evaluation metrics for continuous anomaly scores."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_scores(
    y_true: pd.Series,
    anomaly_scores: pd.Series,
    model_name: str,
) -> dict:
    """Evaluate higher-is-more-anomalous scores and select the best F1 threshold."""
    y = np.asarray(y_true)
    scores = np.asarray(anomaly_scores)
    thresholds = np.unique(np.percentile(scores, np.arange(101)))
    threshold = max(thresholds, key=lambda value: f1_score(y, scores >= value))
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

    return {
        "model_name": model_name,
        "roc_auc": float(roc_auc_score(y, scores)),
        "avg_precision": float(average_precision_score(y, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
