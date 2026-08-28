from pathlib import Path

import pandas as pd

from covertlens.capture.manifest import (
    COLUMNS,
    build_manifest,
    parse_capture_filename,
    summarize_manifest,
    write_manifest,
)


def make_captures(raw_dir: Path) -> None:
    raw_dir.mkdir()
    captures = {
        "dns_baseline_legit_20260829T100000Z.pcap": b"",
        "dns_iodine_covert_20260829T100100Z.pcap": b"abc",
        "icmp_baseline_legit_20260829T100200Z.pcap": b"12",
        "icmp_ptunnel_covert_20260829T100300Z.pcap": b"1234",
        "invalid_name.pcap": b"ignored",
    }
    for filename, content in captures.items():
        (raw_dir / filename).write_bytes(content)


def test_build_manifest_parses_valid_capture_names(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    make_captures(raw_dir)

    manifest = build_manifest(raw_dir)

    assert list(manifest.columns) == COLUMNS
    assert len(manifest) == 4
    assert set(manifest["protocol"]) == {"dns", "icmp"}
    assert set(manifest["source"]) == {"baseline", "iodine", "ptunnel"}
    assert manifest.set_index("source").loc["iodine", "file_size_bytes"] == 3
    assert manifest["packet_count"].isna().all()
    assert parse_capture_filename("invalid_name.pcap") is None


def test_write_manifest_and_summary(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_path = tmp_path / "processed" / "capture_manifest.csv"
    make_captures(raw_dir)

    manifest = write_manifest(raw_dir, output_path)
    summary = summarize_manifest(manifest)

    assert output_path.exists()
    assert len(pd.read_csv(output_path)) == 4
    assert summary.set_index(["protocol", "condition"])["count"].to_dict() == {
        ("dns", "covert"): 1,
        ("dns", "legit"): 1,
        ("icmp", "covert"): 1,
        ("icmp", "legit"): 1,
    }
