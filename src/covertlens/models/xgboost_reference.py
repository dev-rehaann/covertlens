"""Supervised reference, not an unsupervised novelty-detection model.

Labels are deliberately used during training here. A covert-session holdout
tests transfer from OTHER labeled tunnel sessions/tools, a different question
from label-free training. This reference is an upper-bound sanity check in
intent, not a guarantee that it will outperform the unsupervised methods.
"""

import pandas as pd
from xgboost import XGBClassifier


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> XGBClassifier:
    """Fit a binary classifier with negative/positive weighting from this fold.

    Both classes are required: supervised learning does not magically fix a
    training fold with no legitimate examples. Hyperparameters are fixed in
    advance; no test labels or test-session scores guide model selection.
    """
    if X_train.empty or len(X_train) != len(y_train):
        raise ValueError("Training features and labels must be nonempty and equal-length")
    if not X_train.index.equals(y_train.index):
        raise ValueError("Training features and labels must have aligned indices")
    if not y_train.isin([0, 1]).all():
        raise ValueError("Training labels must be 0 (legit) or 1 (covert)")
    negatives = int(y_train.eq(0).sum())
    positives = int(y_train.eq(1).sum())
    if not negatives or not positives:
        raise ValueError("XGBoost reference requires both training classes")

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=1,
        eval_metric="logloss",
        scale_pos_weight=negatives / positives,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def score_xgboost(model: XGBClassifier, X_test: pd.DataFrame) -> pd.Series:
    """Return the positive-class probability; higher means more likely covert."""
    positive_column = list(model.classes_).index(1)
    return pd.Series(
        model.predict_proba(X_test)[:, positive_column],
        index=X_test.index,
        name="covert_probability",
    )
