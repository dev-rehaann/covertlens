"""Presentation regression: preserve evidence labels and mixed window IDs."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("protocol", ["dns", "icmp"])
def test_dashboard_results_and_mixed_flow_ids(protocol):
    results = {
        "run_timestamp": "fixture",
        "per_fold_results": [{"fold": "fixture.pcap", "model_name": "Isolation Forest"}],
        "fpr_summary": [{"model_name": "Isolation Forest", "mean": 0.1, "std": 0.02, "count": 4}],
        "recall_summary": [{"model_name": "Isolation Forest", "mean": 0.8, "std": 0.1, "count": 2}],
    }
    rows = [
        {
            "flow_id": flow_id,
            "start_timestamp": 0.0,
            "end_timestamp": 1.0,
            "packet_count": 2,
            "isolation_forest_score": 0.2,
            "autoencoder_reconstruction_error": 0.4,
            "flagged_by_either": flagged,
        }
        for flow_id, flagged in [(0, False), ("1_w0", True)]
    ]

    def respond(method, url, **kwargs):
        if method == "GET":
            assert url.endswith(f"/results/{protocol}")
            assert kwargs["timeout"] == (5, 10)
        else:
            assert kwargs["params"]["protocol"] == protocol
            assert kwargs["timeout"] == (5, 300)
            assert kwargs["files"] == {
                "file": ("fixture.pcap", b"fixture", "application/octet-stream")
            }
        response = MagicMock()
        response.ok = True
        response.json.return_value = results if method == "GET" else rows
        response.__enter__.return_value = response
        return response

    upload = SimpleNamespace(
        name="fixture.pcap", getbuffer=lambda: b"fixture", getvalue=lambda: b"fixture"
    )
    path = Path(__file__).resolve().parents[1] / "src/covertlens/dashboard/app.py"
    app = AppTest.from_file(str(path), default_timeout=15)
    app.session_state["evaluation_protocol"] = protocol
    app.session_state["scoring_protocol"] = protocol
    with (
        patch("requests.request", side_effect=respond) as request,
        patch("streamlit.file_uploader", return_value=upload),
    ):
        app.run()
        assert not app.exception and not app.error
        assert len(app.tabs) == 2 and len(app.warning) == 1
        assert "100% of available data" in app.warning[0].value
        assert "leave-one-session-out" in app.info[0].value
        assert any("n=4 legit sessions; n=2 covert sessions" in c.value for c in app.caption)
        assert any("Upload received by Streamlit" in c.value for c in app.caption)
        assert any(
            "Flag rate here is not directly comparable to the Evaluation Results tab" in c.value
            and "including all covert sessions" in c.value
            for c in app.caption
        )
        charts = app.get("vega_lite_chart")
        assert len(charts) == 2
        comparison = app.dataframe[0].value
        assert comparison.iloc[0]["FPR"] == "10.00%"
        assert comparison.iloc[0]["Recall"] == "80.00%"
        flows = app.dataframe[-1].value
        assert flows["flow_id"].tolist() == ["0", "1_w0"]
        assert flows["status"].tolist() == ["Not flagged", "Flagged - demo"]
        assert [metric.value for metric in app.metric] == ["2", "1", "4"]
        app.run()
        assert not app.exception
        assert sum(c.args[0] == "POST" for c in request.call_args_list) == 1


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("connection refused"),
        requests.ReadTimeout("read timed out"),
        ValueError("invalid JSON"),
    ],
)
def test_backend_exception_is_visible(error):
    path = Path(__file__).resolve().parents[1] / "src/covertlens/dashboard/app.py"
    request_backend = runpy.run_path(str(path))["request_backend"]
    with (
        patch("requests.request", side_effect=error),
        patch("streamlit.error") as show_error,
        patch("streamlit.caption") as show_caption,
    ):
        assert request_backend("POST", "/score") is None
    message = show_error.call_args.args[0]
    assert type(error).__name__ in message and str(error) in message
    assert "/score" in message
    assert show_caption.called
    if isinstance(error, requests.Timeout):
        assert "wait before retrying" in show_caption.call_args.args[0]


@pytest.mark.parametrize("status", [400, 413, 422, 500])
def test_backend_http_error_is_visible(status):
    path = Path(__file__).resolve().parents[1] / "src/covertlens/dashboard/app.py"
    request_backend = runpy.run_path(str(path))["request_backend"]
    response = MagicMock()
    response.ok = False
    response.status_code = status
    response.json.return_value = {"detail": "fixture failure detail"}
    response.__enter__.return_value = response
    with patch("requests.request", return_value=response), patch("streamlit.error") as show_error:
        assert request_backend("POST", "/score") is None
    assert f"HTTP {status}" in show_error.call_args.args[0]
    assert "fixture failure detail" in show_error.call_args.args[0]
