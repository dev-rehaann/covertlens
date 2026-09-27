"""Presentation regression: preserve evidence labels and mixed window IDs."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
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
        else:
            assert kwargs["params"]["protocol"] == protocol
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
