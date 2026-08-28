import shutil
from pathlib import Path

import pytest


pytest.importorskip("pyshark")
scapy = pytest.importorskip("scapy.all")
if shutil.which("tshark") is None:
    pytest.skip("tshark is required for the PyShark integration test", allow_module_level=True)
parse_dns = pytest.importorskip("covertlens.features.parse_dns")


def test_extract_dns_packet_metadata_from_synthetic_pcap(tmp_path: Path) -> None:
    pcap_path = tmp_path / "synthetic_dns.pcap"
    packets = [
        scapy.IP(src="10.0.0.2", dst="10.0.0.53")
        / scapy.UDP(sport=53000, dport=53)
        / scapy.DNS(rd=1, qd=scapy.DNSQR(qname="example.com", qtype="A")),
        scapy.IP(src="10.0.0.2", dst="10.0.0.53")
        / scapy.UDP(sport=53001, dport=53)
        / scapy.DNS(rd=1, qd=scapy.DNSQR(qname="payload.example.com", qtype="TXT")),
    ]
    scapy.wrpcap(str(pcap_path), packets)

    metadata = parse_dns.extract_dns_packet_metadata(str(pcap_path))

    assert len(metadata) == 2
    assert set(metadata["qtype"]) == {"A", "TXT"}
