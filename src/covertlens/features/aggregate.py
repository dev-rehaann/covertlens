"""Aggregate packet metadata into model-ready flow features."""

from typing import Any

import pandas as pd

from covertlens.features.entropy import shannon_entropy
from covertlens.features.field_anomalies import (
    dns_query_anomaly_score,
    icmp_size_anomaly_score,
)


COLUMNS = [
    "flow_id",
    "protocol",
    "packet_count",
    "duration_seconds",
    "size_mean",
    "size_std",
    "size_cv",
    "interarrival_mean",
    "interarrival_std",
    "interarrival_cv",
    "is_single_packet_flow",
    "entropy_mean",
    "entropy_max",
    "mean_query_length",
    "txt_null_ratio",
    "max_query_length",
    "icmp_size_cv",
]


def _mean_std_cv(values: pd.Series) -> tuple[float, float, float]:
    if values.empty:
        return 0.0, 0.0, 0.0
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return mean, std, std / mean if mean else 0.0


def build_flow_features(packet_df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Return one aggregated feature row per ``flow_id``.

    Packet size, timing, and payload entropy summarize protocol-independent
    side channels. DNS query shape and ICMP payload-size variation add the
    protocol mechanics most associated with each tunnel family.
    """
    protocol = protocol.lower()
    if protocol not in {"dns", "icmp"}:
        raise ValueError("protocol must be 'dns' or 'icmp'")

    rows: list[dict[str, Any]] = []
    for flow_id, flow in packet_df.groupby("flow_id", sort=False):
        flow = flow.sort_values("timestamp")
        size_mean, size_std, size_cv = _mean_std_cv(flow["packet_size"])
        interarrival_mean, interarrival_std, interarrival_cv = _mean_std_cv(
            flow["timestamp"].diff().dropna()
        )
        entropies = flow["payload_bytes"].map(shannon_entropy)

        row = {
            "flow_id": flow_id,
            "protocol": protocol,
            "packet_count": len(flow),
            "duration_seconds": float(flow["timestamp"].max() - flow["timestamp"].min()),
            "size_mean": size_mean,
            "size_std": size_std,
            "size_cv": size_cv,
            "interarrival_mean": interarrival_mean,
            "interarrival_std": interarrival_std,
            "interarrival_cv": interarrival_cv,
            "is_single_packet_flow": len(flow) == 1,
            "entropy_mean": float(entropies.mean()),
            "entropy_max": float(entropies.max()),
            "mean_query_length": float("nan"),
            "txt_null_ratio": float("nan"),
            "max_query_length": float("nan"),
            "icmp_size_cv": float("nan"),
        }

        if protocol == "dns":
            row.update(dns_query_anomaly_score(flow["query_length"], flow["qtype"]))
        else:
            row["icmp_size_cv"] = icmp_size_anomaly_score(flow["payload_length"])
        rows.append(row)

    return pd.DataFrame(rows, columns=COLUMNS)
