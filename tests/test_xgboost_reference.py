import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

reference = pytest.importorskip("covertlens.models.xgboost_reference")
score_xgboost = reference.score_xgboost
train_xgboost = reference.train_xgboost


def test_separable_holdout_and_fold_specific_weight():
    generator = np.random.default_rng(42)
    X = pd.DataFrame(
        np.vstack([generator.normal(0, 0.2, (100, 2)), generator.normal(6, 0.2, (25, 2))]),
        columns=["size", "timing"],
    )
    y = pd.Series([0] * 100 + [1] * 25)
    model = train_xgboost(X, y)
    assert model.scale_pos_weight == pytest.approx(4.0)
    test = pd.DataFrame(
        np.vstack([generator.normal(0, 0.2, (20, 2)), generator.normal(6, 0.2, (10, 2))]),
        columns=X.columns,
        index=range(200, 230),
    )
    scores = score_xgboost(model, test)
    assert scores.index.equals(test.index)
    assert scores.between(0, 1).all()
    assert scores.iloc[-10:].mean() > scores.iloc[:20].mean()
    assert roc_auc_score([0] * 20 + [1] * 10, scores) > 0.95
    subset = list(range(50)) + list(range(100, 125))
    second_fold = train_xgboost(X.iloc[subset], y.iloc[subset])
    assert second_fold.scale_pos_weight == pytest.approx(2.0)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_training_is_rejected(label):
    X = pd.DataFrame({"feature": [0.0, 1.0]})
    with pytest.raises(ValueError, match="both training classes"):
        train_xgboost(X, pd.Series([label, label]))
