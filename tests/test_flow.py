from covertlens.features.flow import assign_flow_id


def dns_packet(timestamp: float, reverse: bool = False) -> dict[str, object]:
    client = ("10.0.0.2", 53000)
    server = ("10.0.0.53", 53)
    src, dst = (server, client) if reverse else (client, server)
    return {
        "timestamp": timestamp,
        "src_ip": src[0],
        "dst_ip": dst[0],
        "src_port": src[1],
        "dst_port": dst[1],
        "protocol": "dns",
    }


def icmp_packet(timestamp: float, src_ip: str, dst_ip: str) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": None,
        "dst_port": None,
        "protocol": "icmp",
    }


def test_close_bidirectional_packets_share_flow() -> None:
    result = assign_flow_id([dns_packet(0), dns_packet(2, reverse=True)])

    assert result["flow_id"].tolist() == [0, 0]


def test_inactivity_timeout_starts_new_flow() -> None:
    result = assign_flow_id([dns_packet(0), dns_packet(31, reverse=True)])

    assert result["flow_id"].tolist() == [0, 1]


def test_interleaved_endpoint_pairs_remain_separate() -> None:
    packets = [
        icmp_packet(0, "10.0.0.2", "10.0.0.10"),
        icmp_packet(1, "10.0.0.3", "10.0.0.11"),
        icmp_packet(2, "10.0.0.10", "10.0.0.2"),
        icmp_packet(3, "10.0.0.11", "10.0.0.3"),
    ]

    result = assign_flow_id(packets)

    assert result["flow_id"].tolist() == [0, 1, 0, 1]
