import pandas as pd
import pytest

from covertlens.models.evaluate import evaluate_scores


def test_perfect_separation_has_unit_auc() -> None:
    result = evaluate_scores(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0.1, 0.2, 0.8, 0.9]),
        "perfect",
    )

    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["avg_precision"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["tp"] == 2
    assert result["tn"] == 2


def test_no_signal_ranking_has_half_auc() -> None:
    result = evaluate_scores(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0.1, 0.4, 0.2, 0.3]),
        "no-signal",
    )

    assert result["roc_auc"] == pytest.approx(0.5)
