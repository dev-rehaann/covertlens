"""Protocol-field anomaly features for DNS and ICMP flows."""

import pandas as pd


def icmp_size_anomaly_score(payload_lengths: pd.Series) -> float:
    """Return the coefficient of variation of ICMP payload sizes in a flow.

    Legitimate OS ping traffic normally repeats one payload size, while a
    covert channel often varies it to pack different amounts of tunneled data.
    Therefore ``standard deviation / mean`` is near zero for a stable ping flow
    and rises as payload sizing becomes anomalous.
    """
    lengths = payload_lengths.dropna()
    if lengths.empty:
        return 0.0

    mean = float(lengths.mean())
    if mean == 0.0:
        return 0.0
    return float(lengths.std(ddof=0) / mean)


def dns_query_anomaly_score(query_lengths: pd.Series, qtypes: pd.Series) -> dict[str, float]:
    """Return DNS length and record-type indicators tied to tunneling mechanics.

    DNS tunnels encode data into query names, making their mean and maximum
    lengths larger than ordinary hostnames. They also use TXT and NULL records
    disproportionately because those types can carry more tunnel data than the
    A/AAAA queries that dominate legitimate DNS traffic.
    """
    lengths = query_lengths.dropna()
    observed_qtypes = qtypes.dropna().astype(str).str.upper()

    return {
        "mean_query_length": float(lengths.mean()) if not lengths.empty else 0.0,
        "txt_null_ratio": (
            float(observed_qtypes.isin({"TXT", "NULL"}).mean())
            if not observed_qtypes.empty
            else 0.0
        ),
        "max_query_length": float(lengths.max()) if not lengths.empty else 0.0,
    }
