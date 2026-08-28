import shutil
from pathlib import Path

import pytest


pytest.importorskip("pyshark")
scapy = pytest.importorskip("scapy.all")
if shutil.which("tshark") is None:
    pytest.skip("tshark is required for the PyShark integration test", allow_module_level=True)
parse_icmp = pytest.importorskip("covertlens.features.parse_icmp")


def test_extract_icmp_packet_metadata_from_synthetic_pcap(tmp_path: Path) -> None:
    pcap_path = tmp_path / "synthetic_icmp.pcap"
    request_payload = b"request-data"
    reply_payload = b"reply-data-longer"
    packets = [
        scapy.IP(src="10.0.0.2", dst="10.0.0.1", ttl=64)
        / scapy.ICMP(type=8, code=0)
        / scapy.Raw(request_payload),
        scapy.IP(src="10.0.0.1", dst="10.0.0.2", ttl=63)
        / scapy.ICMP(type=0, code=0)
        / scapy.Raw(reply_payload),
    ]
    scapy.wrpcap(str(pcap_path), packets)

    metadata = parse_icmp.extract_icmp_packet_metadata(str(pcap_path))

    assert len(metadata) == 2
    assert metadata["icmp_type"].tolist() == [8, 0]
    assert metadata["icmp_code"].tolist() == [0, 0]
    assert metadata["payload_bytes"].tolist() == [request_payload, reply_payload]
    assert metadata["payload_length"].tolist() == [len(request_payload), len(reply_payload)]
    assert metadata["packet_size"].tolist() == [len(bytes(packet)) for packet in packets]
    assert metadata["ttl"].tolist() == [64, 63]
