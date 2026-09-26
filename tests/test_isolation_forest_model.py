import numpy as np
import pandas as pd

from covertlens.models.isolation_forest_model import (
    predict_labels,
    score_flows,
    train_isolation_forest,
)


def test_outliers_receive_higher_anomaly_scores() -> None:
    generator = np.random.default_rng(42)
    cluster = generator.normal(0.0, 0.25, size=(100, 2))
    outliers = np.array([[6.0, 6.0], [-6.0, 6.0], [6.0, -6.0], [-6.0, -6.0], [8.0, 0.0]])
    X = pd.DataFrame(np.vstack([cluster, outliers]), columns=["feature_a", "feature_b"])

    model = train_isolation_forest(X, contamination=5 / len(X))
    scores = score_flows(model, X)
    predictions = predict_labels(model, X)

    assert model.n_estimators == 200
    assert scores.index.equals(X.index)
    assert predictions.index.equals(X.index)
    np.testing.assert_allclose(scores.to_numpy(), -model.decision_function(X))
    np.testing.assert_array_equal(predictions.to_numpy(), (model.predict(X) == -1).astype(int))
    assert scores.iloc[-5:].mean() > scores.iloc[:-5].mean()
    assert scores.iloc[-5:].min() > scores.iloc[:-5].median()
    assert set(predictions) <= {0, 1}
    assert predictions.iloc[-5:].sum() >= 4
