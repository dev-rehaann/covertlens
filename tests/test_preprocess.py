from pathlib import Path

import pandas as pd

from covertlens.models.preprocess import load_and_prepare


def _write_features(path: Path) -> None:
    pd.DataFrame(
        {
            "flow_id": [0, 1, 2, 3],
            "protocol": ["dns", "dns", "icmp", "icmp"],
            "label": [0, 1, 0, 1],
            "source_file": ["dns-a.pcap", "dns-b.pcap", "icmp-a.pcap", "icmp-b.pcap"],
            "packet_count": [1, 3, 1, 4],
            "duration_seconds": [0.0, 2.0, 0.0, 3.0],
            "size_mean": [60.0, 90.0, 80.0, 140.0],
            "size_std": [0.0, 10.0, 0.0, 20.0],
            "size_cv": [0.0, 0.11, 0.0, 0.14],
            "interarrival_mean": [None, 1.0, None, 1.5],
            "interarrival_std": [None, 0.2, None, 0.4],
            "interarrival_cv": [None, 0.2, None, 0.27],
            "entropy_mean": [2.5, 5.5, 2.0, 7.0],
            "entropy_max": [3.0, 6.0, 2.0, 7.5],
            "is_single_packet_flow": [True, False, True, False],
            "mean_query_length": [12.0, 80.0, None, None],
            "txt_null_ratio": [0.0, 0.75, None, None],
            "max_query_length": [16.0, 120.0, None, None],
            "icmp_size_cv": [None, None, 0.0, 0.5],
        }
    ).to_csv(path, index=False)


def test_combined_protocol_preprocessing_fills_structural_nans(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_features(path)

    X, y, feature_names, scaler = load_and_prepare(str(path))

    assert not X.isna().any().any()
    assert (X.mean().abs() < 1e-12).all()
    assert y.tolist() == [0, 1, 0, 1]
    assert not {"label", "source_file", "flow_id", "protocol"}.intersection(feature_names)
    restored = pd.DataFrame(scaler.inverse_transform(X), columns=feature_names, index=X.index)
    assert abs(restored.loc[0, "interarrival_mean"]) < 1e-12
    assert abs(restored.loc[0, "icmp_size_cv"]) < 1e-12
    assert abs(restored.loc[2, "mean_query_length"]) < 1e-12


def test_protocol_filter_drops_irrelevant_columns(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_features(path)

    X, y, feature_names, _ = load_and_prepare(str(path), protocol="icmp")

    assert not X.isna().any().any()
    assert (X.mean().abs() < 1e-12).all()
    assert y.tolist() == [0, 1]
    assert "icmp_size_cv" in feature_names
    assert not {"mean_query_length", "txt_null_ratio", "max_query_length"}.intersection(
        feature_names
    )
