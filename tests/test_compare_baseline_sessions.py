import subprocess

import pandas as pd

from scripts import compare_baseline_sessions as diagnostic


def test_reports_raw_capture_duration_separately(tmp_path, monkeypatch, capsys):
    (tmp_path / "baseline.pcap").touch()
    features = pd.DataFrame({name: [1.0, 3.0] for name in diagnostic.FEATURES})
    features["protocol"] = "dns"
    features["label"] = 0
    features["source_file"] = "baseline.pcap"
    monkeypatch.setattr(diagnostic.shutil, "which", lambda name: "capinfos")
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, "File name,Number of packets,Capture duration (seconds)\nbaseline.pcap,4,600\n"
        ),
    )
    diagnostic.compare_baseline_sessions(features, tmp_path)
    output = capsys.readouterr().out
    assert "600.000000" in output
    assert "raw_duration_seconds" in output
    assert "feature_packet_count_sum" in output
    assert "sample standard deviation" in output
