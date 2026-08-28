import pandas as pd

from covertlens.features.field_anomalies import (
    dns_query_anomaly_score,
    icmp_size_anomaly_score,
)


def test_covert_like_flows_score_higher_than_legit_like_flows() -> None:
    legit_icmp = icmp_size_anomaly_score(pd.Series([56, 56, 56, 56]))
    covert_icmp = icmp_size_anomaly_score(pd.Series([16, 64, 120, 32]))

    legit_dns = dns_query_anomaly_score(
        pd.Series([10, 14, 18, 12]),
        pd.Series(["A", "AAAA", "A", "A"]),
    )
    covert_dns = dns_query_anomaly_score(
        pd.Series([80, 120, 180, 220]),
        pd.Series(["TXT", "TXT", "NULL", "A"]),
    )

    assert legit_icmp == 0.0
    assert covert_icmp > legit_icmp
    assert covert_dns["mean_query_length"] > legit_dns["mean_query_length"]
    assert covert_dns["txt_null_ratio"] > legit_dns["txt_null_ratio"]
    assert covert_dns["max_query_length"] > legit_dns["max_query_length"]


def test_zero_mean_icmp_flow_is_neutral() -> None:
    assert icmp_size_anomaly_score(pd.Series([0, 0, 0])) == 0.0
