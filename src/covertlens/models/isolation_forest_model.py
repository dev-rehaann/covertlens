"""Isolation Forest training and scoring helpers."""

import pandas as pd
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42,
    n_estimators: int = 200,
) -> IsolationForest:
    """Fit an Isolation Forest without using ground-truth labels.

    ``contamination`` is the prior assumption about what fraction of flows is
    anomalous. That fraction is not known in a real deployment, so results are
    sensitive to this setting; this is a genuine study limitation that should
    be reported rather than hidden.
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    return model.fit(X)


def score_flows(model: IsolationForest, X: pd.DataFrame) -> pd.Series:
    """Return anomaly scores where higher means more anomalous.

    Scikit-learn's ``decision_function`` uses the opposite convention: larger
    values indicate more normal samples. Negating it prevents the classic
    silent error of interpreting normality scores as anomaly scores.
    """
    return pd.Series(-model.decision_function(X), index=X.index, name="anomaly_score")


def predict_labels(model: IsolationForest, X: pd.DataFrame) -> pd.Series:
    """Return 0 for normal flows and 1 for anomalous flows."""
    return pd.Series((model.predict(X) == -1).astype(int), index=X.index, name="prediction")
