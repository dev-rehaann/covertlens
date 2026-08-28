import shutil
from pathlib import Path

import pandas as pd
import pytest


pytest.importorskip("pyshark")
scapy = pytest.importorskip("scapy.all")
if shutil.which("tshark") is None:
    pytest.skip("tshark is required for the PyShark integration test", allow_module_level=True)
build_dataset = pytest.importorskip("covertlens.features.build_dataset").build_dataset


def test_build_dataset_from_synthetic_dns_and_icmp_pcaps(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    dns_name = "dns_baseline_legit_20260829T120000Z.pcap"
    icmp_name = "icmp_ptunnel_covert_20260829T120100Z.pcap"

    scapy.wrpcap(
        str(raw_dir / dns_name),
        [
            scapy.IP(src="10.0.0.2", dst="10.0.0.53")
            / scapy.UDP(sport=53000, dport=53)
            / scapy.DNS(rd=1, qd=scapy.DNSQR(qname="example.com", qtype="A"))
        ],
    )
    scapy.wrpcap(
        str(raw_dir / icmp_name),
        [
            scapy.IP(src="10.0.0.2", dst="10.0.0.1", ttl=64)
            / scapy.ICMP(type=8, code=0)
            / scapy.Raw(b"baseline-payload")
        ],
    )

    manifest_path = tmp_path / "capture_manifest.csv"
    output_path = tmp_path / "features.csv"
    pd.DataFrame(
        [
            {
                "filename": dns_name,
                "protocol": "dns",
                "source": "baseline",
                "condition": "legit",
                "timestamp": "20260829T120000Z",
                "file_size_bytes": (raw_dir / dns_name).stat().st_size,
                "packet_count": None,
            },
            {
                "filename": icmp_name,
                "protocol": "icmp",
                "source": "ptunnel",
                "condition": "covert",
                "timestamp": "20260829T120100Z",
                "file_size_bytes": (raw_dir / icmp_name).stat().st_size,
                "packet_count": None,
            },
        ]
    ).to_csv(manifest_path, index=False)

    features = build_dataset(manifest_path, raw_dir, output_path)

    assert output_path.is_file()
    assert len(features) == 2
    assert set(features["protocol"]) == {"dns", "icmp"}
    assert set(features["label"]) == {0, 1}
    assert features["packet_count"].tolist() == [1, 1]
    assert (features["duration_seconds"] == 0.0).all()
    assert (features["size_mean"] > 0.0).all()
    assert set(features["source_file"]) == {dns_name, icmp_name}
