import math
import shutil
from pathlib import Path

import joblib
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from covertlens.dashboard import api
from covertlens.features.aggregate import build_flow_features
from covertlens.models.train_final import DEMO_WARNING, train_final


@pytest.fixture(scope="module")
def final_models(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("api_models")
    frames = []
    for protocol in ("dns", "icmp"):
        packets = pd.DataFrame(
            {
                "flow_id": [0, 1],
                "timestamp": [10.0, 11.0],
                "packet_size": [60, 140],
                "payload_bytes": [b"example.com", bytes(range(56))],
                "query_length": [11, 24],
                "qtype": ["A", "TXT"],
                "payload_length": [11, 56],
            }
        )
        frame = build_flow_features(packets, protocol)
        frame["label"] = [0, 1]
        frame["source_file"] = [f"{protocol}-baseline.pcap", f"{protocol}-covert.pcap"]
        frames.append(frame)
    features_path = directory / "features.csv"
    pd.concat(frames, ignore_index=True).to_csv(features_path, index=False)
    output_dir = directory / "models"
    train_final(str(features_path), output_dir)
    return output_dir


def test_health_and_protocol_validation():
    with TestClient(api.app) as client:
        assert client.get("/").json()["status"] == "ok"
        assert client.get("/").json()["live_scoring_warning"] == DEMO_WARNING
        assert client.get("/results/tcp").status_code == 422
        assert (
            client.post("/score?protocol=tcp", files={"file": ("test.pcap", b"")}).status_code
            == 422
        )


def test_results_latest_run_summary_and_json_nulls(tmp_path, monkeypatch):
    path = tmp_path / "loso_results.csv"
    rows = [
        ("dns", "old", "2026-01-01T00:00:00Z", "legit-holdout", 0.9, None),
        ("dns", "baseline-1", "2026-09-26T00:00:00Z", "legit-holdout", 0.1, None),
        ("dns", "baseline-2", "2026-09-26T00:00:00Z", "legit-holdout", 0.3, None),
        ("dns", "covert-1", "2026-09-26T00:00:00Z", "covert-holdout", None, 0.8),
        ("icmp", "icmp-1", "2026-09-26T00:00:00Z", "legit-holdout", 0.0, None),
    ]
    table = pd.DataFrame(
        rows, columns=["protocol", "fold", "run_timestamp", "fold_type", "fpr", "recall"]
    )
    table["model_name"] = "Isolation Forest"
    table["roc_auc"] = float("nan")
    table.to_csv(path, index=False)
    monkeypatch.setattr(api, "RESULTS_PATH", path)
    with TestClient(api.app) as client:
        response = client.get("/results/dns")
    assert response.status_code == 200
    body = response.json()
    assert body["evaluation"] == "leave-one-session-out"
    assert body["run_timestamp"] == "2026-09-26T00:00:00Z"
    assert len(body["per_fold_results"]) == 3
    assert all(
        row["protocol"] == "dns" and row["roc_auc"] is None for row in body["per_fold_results"]
    )
    assert body["fpr_summary"][0]["mean"] == pytest.approx(0.2)
    assert body["fpr_summary"][0]["std"] == pytest.approx(math.sqrt(0.02))
    assert body["fpr_summary"][0]["count"] == 2
    assert body["recall_summary"][0] == {
        "model_name": "Isolation Forest",
        "mean": 0.8,
        "std": None,
        "count": 1,
    }


def test_missing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RESULTS_PATH", tmp_path / "missing.csv")
    with TestClient(api.app) as client:
        response = client.get("/results/dns")
    assert response.status_code == 404
    assert "run_loso_evaluation" in response.json()["detail"]


@pytest.mark.parametrize("filename", ["traffic.txt", "model.joblib", "traffic.pcap.exe"])
def test_score_rejects_wrong_extension(filename, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELS_PATH", tmp_path / "missing_models")
    with TestClient(api.app) as client:
        response = client.post("/score?protocol=dns", files={"file": (filename, b"bad")})
    assert response.status_code == 400
    assert ".pcap or .pcapng" in response.json()["detail"]


def test_score_missing_models(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELS_PATH", tmp_path / "missing_models")
    with TestClient(api.app) as client:
        response = client.post("/score?protocol=dns", files={"file": ("traffic.pcap", b"bad")})
    assert response.status_code == 400
    assert "train_final" in response.json()["detail"]


def test_score_stale_thresholdless_models(final_models, tmp_path, monkeypatch):
    copy = tmp_path / "stale_models"
    shutil.copytree(final_models, copy)
    forest_path = copy / "dns_isolation_forest.joblib"
    forest = joblib.load(forest_path)
    del forest.anomaly_threshold_
    joblib.dump(forest, forest_path)
    monkeypatch.setattr(api, "MODELS_PATH", copy)
    with TestClient(api.app) as client:
        response = client.post("/score?protocol=dns", files={"file": ("traffic.pcap", b"bad")})
    assert response.status_code == 400
    assert "thresholds" in response.json()["detail"]


def test_score_malformed_capture_and_cleanup(final_models, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELS_PATH", final_models)
    monkeypatch.setattr(api.tempfile, "tempdir", str(tmp_path))
    with TestClient(api.app) as client:
        response = client.post("/score?protocol=dns", files={"file": ("traffic.pcap", b"bad")})
    assert response.status_code == 400
    assert "Malformed capture" in response.json()["detail"]
    assert not list(tmp_path.glob("covertlens_upload_*"))


def test_score_upload_limit_and_cleanup(final_models, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELS_PATH", final_models)
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    monkeypatch.setattr(api.tempfile, "tempdir", str(tmp_path))
    with TestClient(api.app) as client:
        response = client.post("/score?protocol=dns", files={"file": ("traffic.pcap", b"12345")})
    assert response.status_code == 413
    assert not list(tmp_path.glob("covertlens_upload_*"))


@pytest.mark.parametrize(
    "protocol,suffix", [("dns", ".pcap"), ("icmp", ".pcap"), ("dns", ".pcapng")]
)
def test_score_synthetic_pcap(protocol, suffix, final_models, tmp_path, monkeypatch):
    scapy = pytest.importorskip("scapy.all")
    if shutil.which("tshark") is None:
        pytest.skip("tshark is required for the full scoring integration test")
    packets = []
    if protocol == "dns":
        for index, qtype in enumerate(("A", "TXT")):
            packet = scapy.IP(src="10.0.0.2", dst="10.0.0.53") / scapy.UDP(
                sport=53000 + index, dport=53
            )
            packet /= scapy.DNS(rd=1, qd=scapy.DNSQR(qname="example.com", qtype=qtype))
            packet.time = 1000.0 + index * 2
            packets.append(packet)
    else:
        for index in range(10):
            src, dst = ("10.0.0.2", "10.0.0.1") if index % 2 == 0 else ("10.0.0.1", "10.0.0.2")
            packet = (
                scapy.IP(src=src, dst=dst)
                / scapy.ICMP(type=8 if index % 2 == 0 else 0)
                / scapy.Raw(b"lab payload")
            )
            packet.time = 1000.0 + index * 10
            packets.append(packet)
    path = tmp_path / f"synthetic_{protocol}{suffix}"
    if suffix == ".pcapng":
        with scapy.PcapNgWriter(str(path)) as writer:
            writer.write(packets)
    else:
        scapy.wrpcap(str(path), packets)
    monkeypatch.setattr(api, "MODELS_PATH", final_models)
    seen_paths = []
    original_extractor = api.EXTRACTORS[protocol]

    def track_extraction(filename):
        seen_paths.append(Path(filename))
        return original_extractor(filename)

    monkeypatch.setitem(api.EXTRACTORS, protocol, track_extraction)
    with TestClient(api.app) as client:
        response = client.post(
            f"/score?protocol={protocol}", files={"file": (path.name, path.read_bytes())}
        )
    assert response.status_code == 200, response.text
    assert response.headers["X-Covertlens-Evidence"] == "full-data-live-demo-not-LOSO"
    assert response.headers["X-Covertlens-Warning"] == DEMO_WARNING
    body = response.json()
    assert len(body) == (2 if protocol == "dns" else 3)
    assert sum(row["packet_count"] for row in body) == len(packets)
    forest = joblib.load(final_models / f"{protocol}_isolation_forest.joblib")
    checkpoint = torch.load(final_models / f"{protocol}_autoencoder.pt", weights_only=True)
    for row in body:
        assert set(row) == {
            "flow_id",
            "start_timestamp",
            "end_timestamp",
            "packet_count",
            "isolation_forest_score",
            "autoencoder_reconstruction_error",
            "flagged_by_either",
        }
        assert row["start_timestamp"] <= row["end_timestamp"]
        assert math.isfinite(row["isolation_forest_score"])
        assert math.isfinite(row["autoencoder_reconstruction_error"])
        assert row["flagged_by_either"] == (
            row["isolation_forest_score"] >= forest.anomaly_threshold_
            or row["autoencoder_reconstruction_error"] >= checkpoint["anomaly_threshold"]
        )
    if protocol == "icmp":
        assert [row["start_timestamp"] for row in body] == [1000.0, 1030.0, 1060.0]
        assert [row["end_timestamp"] for row in body] == [1020.0, 1050.0, 1090.0]
    assert seen_paths and not seen_paths[0].exists() and not seen_paths[0].parent.exists()
    with TestClient(api.app) as client:
        truncated = client.post(
            f"/score?protocol={protocol}",
            files={"file": (path.name, path.read_bytes()[:-5])},
        )
    assert truncated.status_code == 400
    assert not seen_paths[-1].exists() and not seen_paths[-1].parent.exists()
    if protocol == "dns" and suffix == ".pcap":
        monkeypatch.setattr(
            api, "reconstruction_error", lambda model, X: pd.Series(float("inf"), index=X.index)
        )
        with TestClient(api.app) as client:
            invalid_scores = client.post(
                "/score?protocol=dns", files={"file": (path.name, path.read_bytes())}
            )
        assert invalid_scores.status_code == 400
        assert "out of range" in invalid_scores.json()["detail"]
        assert not seen_paths[-1].parent.exists()
    if protocol == "icmp":
        with TestClient(api.app) as client:
            wrong_protocol = client.post(
                "/score?protocol=dns", files={"file": (path.name, path.read_bytes())}
            )
        assert wrong_protocol.status_code == 400
        assert "no usable dns packets" in wrong_protocol.json()["detail"]
