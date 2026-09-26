import numpy as np
import pandas as pd
import pytest

from covertlens.models.evaluate import evaluate_scores, evaluate_scores_no_threshold


def test_perfect_separation_has_unit_auc() -> None:
    result = evaluate_scores(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0.1, 0.2, 0.8, 0.9]),
        "perfect",
        training_scores=pd.Series([0.1, 0.2, 0.3, 0.4]),
    )

    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["avg_precision"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["tp"] == 2
    assert result["tn"] == 2
    assert result["fp"] == result["fn"] == 0
    assert result["precision"] == result["recall"] == pytest.approx(1.0)
    assert result["model_name"] == "perfect"


def test_no_signal_ranking_has_half_auc() -> None:
    result = evaluate_scores_no_threshold(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0.1, 0.4, 0.2, 0.3]),
        "no-signal",
    )

    assert result["roc_auc"] == pytest.approx(0.5)
    assert set(result) == {"model_name", "roc_auc", "avg_precision"}


def test_threshold_depends_only_on_training_scores() -> None:
    scores = pd.Series([0.1, 0.4, 0.2, 0.3])
    train = pd.Series([0.0, 0.1, 0.2, 0.3])
    first = evaluate_scores(pd.Series([0, 0, 1, 1]), scores, "first", training_scores=train)
    second = evaluate_scores(pd.Series([1, 1, 0, 0]), scores + 100, "second", training_scores=train)
    assert first["threshold"] == second["threshold"] == pytest.approx(0.27)
    assert first["tp"] + first["fp"] + first["tn"] + first["fn"] == len(scores)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_session_metrics_are_unavailable(label: int) -> None:
    result = evaluate_scores(
        pd.Series([label, label]),
        pd.Series([0.1, 0.9]),
        "single-class",
        training_scores=pd.Series([0.2, 0.3]),
    )
    assert np.isnan(result["roc_auc"]) and np.isnan(result["avg_precision"])
    assert result["tp"] + result["fp"] + result["tn"] + result["fn"] == 2


def test_fixed_probability_threshold_is_not_fit_to_test_labels():
    scores = pd.Series([0.2, 0.7])
    first = evaluate_scores(pd.Series([0, 1]), scores, "reference", threshold=0.5)
    second = evaluate_scores(pd.Series([1, 0]), scores, "reference", threshold=0.5)
    assert first["threshold"] == second["threshold"] == 0.5
    assert first["threshold_source"] == "fixed"
    assert first["tp"] == first["tn"] == 1
    assert np.isnan(first["contamination"])
    with pytest.raises(ValueError, match="without training_scores"):
        evaluate_scores(
            pd.Series([0, 1]), scores, "reference", threshold=0.5, training_scores=scores
        )
