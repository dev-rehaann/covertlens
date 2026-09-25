import pandas as pd

from covertlens.features.aggregate import build_flow_features
from covertlens.features.windowing import split_long_flow


def test_splits_long_flow_and_preserves_short_flow() -> None:
    long_flow = pd.DataFrame(
        {
            "flow_id": "flow_12",
            "timestamp": [float(timestamp) for timestamp in range(0, 151, 2)],
        }
    )

    windows = split_long_flow(long_flow, window_seconds=30.0)

    assert len(windows) == 5
    assert [len(window) for window in windows] == [15, 15, 15, 15, 16]
    assert all(
        window["timestamp"].max() - window["timestamp"].min() <= 30.0
        for window in windows
    )

    short_flow = long_flow[long_flow["timestamp"] <= 10.0]
    short_result = split_long_flow(short_flow)
    assert len(short_result) == 1
    assert short_result[0] is short_flow


def test_aggregate_suffixes_long_flow_windows() -> None:
    packets = pd.DataFrame(
        {
            "flow_id": "flow_12",
            "timestamp": [float(timestamp) for timestamp in range(0, 151, 2)],
            "packet_size": 84,
            "payload_bytes": [b"payload"] * 76,
            "payload_length": 56,
        }
    )

    features = build_flow_features(packets, "icmp")

    assert features["flow_id"].tolist() == [
        f"flow_12_w{index}" for index in range(5)
    ]
