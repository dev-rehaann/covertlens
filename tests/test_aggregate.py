import pandas as pd

from covertlens.features.aggregate import build_flow_features


def test_build_flow_features_for_dns_and_icmp() -> None:
    packets = pd.DataFrame(
        [
            {
                "flow_id": "dns-1",
                "protocol": "dns",
                "timestamp": 10.0,
                "packet_size": 60,
                "payload_bytes": b"example.com",
                "query_length": 11,
                "qtype": "A",
                "payload_length": None,
            },
            {
                "flow_id": "icmp-1",
                "protocol": "icmp",
                "timestamp": 20.0,
                "packet_size": 100,
                "payload_bytes": b"a" * 72,
                "query_length": None,
                "qtype": None,
                "payload_length": 72,
            },
            {
                "flow_id": "dns-1",
                "protocol": "dns",
                "timestamp": 12.0,
                "packet_size": 80,
                "payload_bytes": b"payload.example.com",
                "query_length": 19,
                "qtype": "TXT",
                "payload_length": None,
            },
            {
                "flow_id": "icmp-1",
                "protocol": "icmp",
                "timestamp": 25.0,
                "packet_size": 140,
                "payload_bytes": b"b" * 112,
                "query_length": None,
                "qtype": None,
                "payload_length": 112,
            },
        ]
    )

    dns = build_flow_features(packets[packets["protocol"] == "dns"], "dns")
    icmp = build_flow_features(packets[packets["protocol"] == "icmp"], "icmp")
    features = pd.concat([dns, icmp], ignore_index=True)

    assert len(features) == 2
    assert features.set_index("flow_id")["packet_count"].to_dict() == {
        "dns-1": 2,
        "icmp-1": 2,
    }
    assert dns.loc[0, "size_mean"] == 70.0
    assert dns.loc[0, "duration_seconds"] == 2.0
    assert icmp.loc[0, "size_mean"] == 120.0
    assert icmp.loc[0, "duration_seconds"] == 5.0
    assert pd.isna(dns.loc[0, "icmp_size_cv"])
    assert pd.isna(icmp.loc[0, "mean_query_length"])

    single = build_flow_features(packets.iloc[[0]], "dns").iloc[0]
    assert bool(single["is_single_packet_flow"])
    assert single["interarrival_mean"] == single["interarrival_std"] == 0.0
