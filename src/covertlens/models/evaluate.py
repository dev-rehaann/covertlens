"""Continuous-score metrics with training-calibrated or predeclared thresholds."""

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


def evaluate_scores_no_threshold(
    y_true: pd.Series, anomaly_scores: pd.Series, model_name: str
) -> dict:
    """Return continuous-score ROC-AUC/AP without selecting a threshold.

    Single-class sessions cannot measure discrimination: ROC-AUC is undefined,
    and AP is trivially 1 for all-positive data (or uninformative with no
    positives). Return NaN for both in that case, excluding them from fold
    means rather than claiming perfect or failed discrimination.
    """
    y = np.asarray(y_true)
    scores = np.asarray(anomaly_scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Labels and scores must be nonempty, equal-length vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be 0/1 and anomaly scores must be finite")

    both_classes = len(np.unique(y)) == 2
    return {
        "model_name": model_name,
        "roc_auc": float(roc_auc_score(y, scores)) if both_classes else float("nan"),
        "avg_precision": (
            float(average_precision_score(y, scores)) if both_classes else float("nan")
        ),
    }


def evaluate_scores(
    y_true: pd.Series,
    anomaly_scores: pd.Series,
    model_name: str,
    *,
    training_scores: pd.Series | None = None,
    contamination: float = 0.1,
    threshold: float | None = None,
) -> dict:
    """Evaluate using the training-score (1-contamination) percentile threshold.

    Higher scores mean more anomalous. Neither evaluation labels nor test
    scores calibrate the threshold. Apply the same anomaly-fraction prior to
    both model families. Always return the full confusion matrix, including
    when the held-out session contains a single class.

    A predeclared fixed threshold is also supported for supervised probability
    scores (0.5 for the XGBoost reference). It must not be chosen from test
    labels/scores; do not pass training_scores together with a fixed threshold.
    """
    metrics = evaluate_scores_no_threshold(y_true, anomaly_scores, model_name)
    if threshold is None:
        training = np.asarray(training_scores, dtype=float)
        if training.ndim != 1 or not len(training) or not np.isfinite(training).all():
            raise ValueError("training_scores must be a nonempty finite vector")
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        threshold = float(np.percentile(training, 100 * (1 - contamination)))
        threshold_source = "training_percentile"
    else:
        if training_scores is not None or not np.isfinite(threshold):
            raise ValueError("A fixed finite threshold must be provided without training_scores")
        threshold = float(threshold)
        threshold_source = "fixed"
    predictions = (np.asarray(anomaly_scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        **metrics,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "contamination": (
            contamination if threshold_source == "training_percentile" else float("nan")
        ),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
