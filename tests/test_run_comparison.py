import sys

import numpy as np
import pandas as pd

from covertlens.models import run_comparison


def test_comparison_preserves_protocol_runs_and_scales_training_only(tmp_path, monkeypatch):
    features_path = tmp_path / "features.csv"
    results_path = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "protocol": ["dns"] * 20 + ["icmp"] * 20,
            "label": [0, 1] * 20,
            "source_file": ["dns.pcap"] * 20 + ["icmp.pcap"] * 20,
            "is_single_packet_flow": False,
            "size_mean": np.arange(40, dtype=float) ** 2,
        }
    ).to_csv(features_path, index=False)
    monkeypatch.setattr(run_comparison, "FEATURES_PATH", features_path)
    monkeypatch.setattr(run_comparison, "RESULTS_PATH", results_path)

    def fake_train(X):
        np.testing.assert_allclose(X.mean(), 0.0, atol=1e-12)
        return object()

    def fake_scores(model, X):
        return X["size_mean"]

    monkeypatch.setattr(run_comparison, "train_isolation_forest", fake_train)
    monkeypatch.setattr(run_comparison, "train_autoencoder", fake_train)
    monkeypatch.setattr(run_comparison, "score_flows", fake_scores)
    monkeypatch.setattr(run_comparison, "reconstruction_error", fake_scores)

    for protocol in ("dns", "icmp", None):
        arguments = ["run_comparison"]
        if protocol:
            arguments.extend(["--protocol", protocol])
        monkeypatch.setattr(sys, "argv", arguments)
        run_comparison.main()

    results = pd.read_csv(results_path)
    assert len(results) == 6
    assert results["protocol"].tolist() == ["dns"] * 2 + ["icmp"] * 2 + ["combined"] * 2
    assert results["run_timestamp"].notna().all()
