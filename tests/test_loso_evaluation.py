import sys

import numpy as np
import pandas as pd

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
    run_loso_evaluation.main()

    results = pd.read_csv(results_path)
    assert results["fold"].nunique() == 3
    assert results.groupby("fold").size().tolist() == [2, 2, 2]
    assert sorted(holdouts) == ["baseline.pcap"] * 2 + ["hans.pcap"] * 2 + ["ptunnel.pcap"] * 2
    assert results[["roc_auc", "avg_precision"]].isna().all().all()
    assert (results[["tp", "fp", "tn", "fn"]].sum(axis=1) == 3).all()
