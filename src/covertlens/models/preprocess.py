"""Prepare flow features for unsupervised models without using labels as inputs."""

import pandas as pd
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = ["label", "source_file", "flow_id", "protocol"]
DNS_COLUMNS = ["mean_query_length", "txt_null_ratio", "max_query_length"]
ICMP_COLUMNS = ["icmp_size_cv"]
INTERARRIVAL_COLUMNS = ["interarrival_mean", "interarrival_std", "interarrival_cv"]


def load_and_prepare(
    features_csv_path: str,
    protocol: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], StandardScaler]:
    """Load, filter, impute, and standardize model features.

    Labels are returned separately for evaluation only and are never included
    in the unsupervised model inputs. Missing interarrival values on a
    single-packet flow become zero because that flow has no timing variance to
    measure; this is a conservative representation rather than invented data.
    """
    features = pd.read_csv(features_csv_path)
    missing = [column for column in ("protocol", "label") if column not in features]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if protocol is not None:
        protocol = protocol.lower()
        if protocol not in {"dns", "icmp"}:
            raise ValueError("protocol must be 'dns', 'icmp', or None")
        features = features[features["protocol"].str.lower() == protocol].copy()
        if features.empty:
            raise ValueError(f"No rows found for protocol '{protocol}'")

    y = pd.to_numeric(features["label"], errors="raise")
    X = features.drop(columns=METADATA_COLUMNS, errors="ignore")
    if protocol == "dns":
        X = X.drop(columns=ICMP_COLUMNS, errors="ignore")
    elif protocol == "icmp":
        X = X.drop(columns=DNS_COLUMNS, errors="ignore")

    single_packet = features["is_single_packet_flow"].isin([True, 1, "True", "true"])
    timing_columns = [column for column in INTERARRIVAL_COLUMNS if column in X]
    X.loc[single_packet, timing_columns] = X.loc[single_packet, timing_columns].fillna(0.0)

    if protocol is None:
        # Cross-protocol zero-fill is a simple structural-missingness encoding;
        # revisit it if a protocol-aware imputer improves Phase 3 evaluation.
        protocol_columns = [column for column in [*DNS_COLUMNS, *ICMP_COLUMNS] if column in X]
        X[protocol_columns] = X[protocol_columns].fillna(0.0)

    X = X.apply(pd.to_numeric, errors="raise")
    if X.isna().any().any():
        columns = ", ".join(X.columns[X.isna().any()])
        raise ValueError(f"Unhandled NaN values remain in: {columns}")

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_names,
        index=features.index,
    )
    return X, y, feature_names, scaler
