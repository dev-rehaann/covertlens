import sys

import numpy as np
import pandas as pd
import pytest

from covertlens.models import run_loso_evaluation


def test_three_folds_keep_sessions_disjoint_and_scale_train_only(tmp_path, monkeypatch):
    features = pd.DataFrame(
        {
            "protocol": "icmp",
            "label": [0, 0, 0, 1, 1, 1, 1, 1, 1],
            "source_file": ["baseline.pcap"] * 3 + ["hans.pcap"] * 3 + ["ptunnel.pcap"] * 3,
            "is_single_packet_flow": False,
            "size_mean": np.arange(9, dtype=float) ** 2,
        }
    )
    features_path = tmp_path / "features.csv"
    results_path = tmp_path / "results.csv"
    features.to_csv(features_path, index=False)
    monkeypatch.setattr(run_loso_evaluation, "FEATURES_PATH", features_path)
    monkeypatch.setattr(run_loso_evaluation, "RESULTS_PATH", results_path)
    monkeypatch.setattr(sys, "argv", ["run_loso_evaluation", "--protocol", "icmp"])
    holdouts = []

    def fake_train(X, **kwargs):
        np.testing.assert_allclose(X.mean(), 0, atol=1e-12)
        return set(features.loc[X.index, "source_file"])

    def fake_score(training_groups, X):
        scoring_groups = set(features.loc[X.index, "source_file"])
        if scoring_groups != training_groups:
            assert len(scoring_groups) == 1
            assert scoring_groups.isdisjoint(training_groups)
            holdouts.append(next(iter(scoring_groups)))
        return X["size_mean"]

    monkeypatch.setattr(run_loso_evaluation, "train_isolation_forest", fake_train)
    monkeypatch.setattr(run_loso_evaluation, "train_autoencoder", fake_train)
    monkeypatch.setattr(run_loso_evaluation, "score_flows", fake_score)
    monkeypatch.setattr(run_loso_evaluation, "reconstruction_error", fake_score)

    def fake_supervised_train(X, y):
        assert y.equals(features.loc[X.index, "label"])
        assert set(y) == {0, 1}
        return fake_train(X)

    monkeypatch.setattr(run_loso_evaluation, "train_xgboost", fake_supervised_train)
    monkeypatch.setattr(run_loso_evaluation, "score_xgboost", fake_score, raising=False)
    run_loso_evaluation.main()

    results = pd.read_csv(results_path)
    assert results["fold"].nunique() == 3
    assert results.groupby("fold").size().tolist() == [2, 3, 3]
    assert sorted(holdouts) == ["baseline.pcap"] * 2 + ["hans.pcap"] * 3 + ["ptunnel.pcap"] * 3
    supervised = results.loc[results["model_name"] == run_loso_evaluation.XGBOOST_NAME]
    assert supervised["uses_training_labels"].all()
    assert supervised["scale_pos_weight"].eq(1.0).all()
    assert supervised["threshold"].eq(0.5).all()
    assert supervised["threshold_source"].eq("fixed").all()
    assert results[["roc_auc", "avg_precision"]].isna().all().all()
    assert (results[["tp", "fp", "tn", "fn"]].sum(axis=1) == 3).all()
    legit = results.loc[results["fold"] == "baseline.pcap"]
    covert = results.loc[results["fold"] != "baseline.pcap"]
    assert legit["fold_type"].eq("legit-holdout").all()
    assert legit["fpr"].notna().all() and legit["recall"].isna().all()
    assert legit["training_had_no_legit_examples"].all()
    assert covert["fold_type"].eq("covert-holdout").all()
    assert covert["recall"].notna().all() and covert["fpr"].isna().all()
    assert not covert["training_had_no_legit_examples"].any()
    for row in legit.itertuples():
        assert row.fpr == pytest.approx(row.fp / (row.fp + row.tn))
    for row in covert.itertuples():
        assert row.recall == pytest.approx(row.tp / (row.tp + row.fn))
