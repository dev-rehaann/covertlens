import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from covertlens.models.autoencoder_model import FlowAutoencoder, reconstruction_error
from covertlens.models.isolation_forest_model import score_flows
from covertlens.models.preprocess import load_and_prepare
from covertlens.models.train_final import DEMO_WARNING, train_final


def test_final_artifacts_round_trip_for_both_protocols(tmp_path: Path, capsys) -> None:
    features_path = tmp_path / "features.csv"
    output_dir = tmp_path / "models_release"
    frame = pd.DataFrame(
        {
            "flow_id": [0, 1, 2, 3],
            "protocol": ["dns", "dns", "icmp", "icmp"],
            "label": [0, 1, 0, 1],
            "source_file": ["dns-a.pcap", "dns-b.pcap", "icmp-a.pcap", "icmp-b.pcap"],
            "size_mean": [60.0, 90.0, 80.0, 140.0],
            "packet_count": [1, 3, 1, 4],
            "is_single_packet_flow": [True, False, True, False],
            "interarrival_mean": [None, 1.0, None, 1.5],
            "interarrival_std": [None, 0.2, None, 0.4],
            "interarrival_cv": [None, 0.2, None, 0.27],
            "mean_query_length": [12.0, 80.0, None, None],
            "max_query_length": [16.0, 120.0, None, None],
            "txt_null_ratio": [0.0, 0.75, None, None],
            "icmp_size_cv": [None, None, 0.0, 0.5],
        }
    )
    frame.to_csv(features_path, index=False)

    train_final(str(features_path), output_dir)

    expected_files = {
        f"{protocol}_{suffix}"
        for protocol in ("dns", "icmp")
        for suffix in (
            "isolation_forest.joblib",
            "autoencoder.pt",
            "scaler.joblib",
            "feature_names.json",
        )
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    for protocol in ("dns", "icmp"):
        X, _, expected_names, original_scaler, _ = load_and_prepare(
            str(features_path), protocol=protocol
        )
        names = json.loads((output_dir / f"{protocol}_feature_names.json").read_text())
        scaler = joblib.load(output_dir / f"{protocol}_scaler.joblib")
        forest = joblib.load(output_dir / f"{protocol}_isolation_forest.joblib")
        checkpoint = torch.load(output_dir / f"{protocol}_autoencoder.pt", weights_only=True)

        assert names == expected_names == list(scaler.feature_names_in_)
        assert scaler.n_samples_seen_ == checkpoint["training_flow_count"] == len(X) == 2
        assert forest.n_features_in_ == scaler.n_features_in_ == len(names)
        assert checkpoint["input_dim"] == len(names)
        assert checkpoint["demo_only"] is True
        assert checkpoint["warning"] == DEMO_WARNING
        assert "max_query_length" not in names
        raw = pd.DataFrame(original_scaler.inverse_transform(X), columns=names, index=X.index)
        restored_X = pd.DataFrame(scaler.transform(raw), columns=names, index=X.index)
        np.testing.assert_allclose(restored_X, X)
        assert np.isfinite(score_flows(forest, restored_X)).all()

        autoencoder = FlowAutoencoder(checkpoint["input_dim"])
        autoencoder.load_state_dict(checkpoint["state_dict"])
        autoencoder.eval()
        errors = reconstruction_error(autoencoder, restored_X)
        assert len(errors) == len(X)
        assert np.isfinite(errors).all() and errors.ge(0).all()

    stdout = capsys.readouterr().out
    assert stdout.count(DEMO_WARNING) == 4
    assert "on 2 total dns flows" in stdout
    assert "on 2 total icmp flows" in stdout
