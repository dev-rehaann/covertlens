"""Local results viewer; full-data live-demo scores are not LOSO evidence.

Run with: python -m streamlit run src/covertlens/dashboard/app.py
Only upload authorized captures. The backend handles extraction and scoring.
"""

import hashlib
import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("COVERTLENS_API_URL", "http://localhost:8000").rstrip("/")
EVALUATION_CAPTION = (
    "These results use leave-one-session-out cross-validation, evaluated on real lab captures. "
    "See docs/notes.md for full methodology and limitations."
)
DEMO_WARNING = (
    "Demo only. This model is trained on 100% of available data with nothing held out, "
    "and has NOT been evaluated with the same rigor as the Evaluation Results tab. "
    "Do not treat its output as equivalent evidence."
)


def request_backend(method: str, path: str, **kwargs):
    """Show API errors inline without preventing the other tab from rendering."""
    try:
        with requests.request(
            method, f"{API_BASE_URL}{path}", timeout=(5, 300), **kwargs
        ) as response:
            if not response.ok:
                try:
                    detail = response.json().get("detail", response.reason)
                except (ValueError, AttributeError):
                    detail = response.reason
                st.error(f"Backend returned HTTP {response.status_code}: {detail}")
                return None
            return response.json()
    except (requests.RequestException, ValueError):
        st.error(
            f"Could not get a valid response from {API_BASE_URL}. "
            "Check that the FastAPI backend is running, then retry."
        )
        return None


def evaluation_results():
    st.info(EVALUATION_CAPTION)
    protocol = st.selectbox(
        "Evaluation protocol", ["dns", "icmp"], format_func=str.upper, key="evaluation_protocol"
    )
    st.button("Refresh results")
    with st.spinner("Loading saved LOSO results..."):
        results = request_backend("GET", f"/results/{protocol}")
    if results is None:
        return

    st.caption(f"Saved run: {results.get('run_timestamp') or 'not recorded'}")
    st.subheader("Per-session held-out results")
    st.dataframe(pd.DataFrame(results["per_fold_results"]), hide_index=True, width="stretch")

    summaries = pd.concat(
        [
            pd.DataFrame(results["fpr_summary"]).assign(metric="FPR"),
            pd.DataFrame(results["recall_summary"]).assign(metric="Recall"),
        ],
        ignore_index=True,
    )
    st.subheader("Session-level FPR and recall")
    if summaries.empty:
        st.info("No FPR/recall summaries are available for this run.")
        return
    st.caption(
        "Lower FPR is better; higher recall is better. "
        "Counts refer to held-out sessions, not correlated flow windows."
    )
    for model_name, rows in summaries.groupby("model_name", sort=False):
        counts = rows.set_index("metric")["count"]
        st.caption(
            f"{model_name}: n={int(counts.get('FPR', 0))} legit sessions; "
            f"n={int(counts.get('Recall', 0))} covert sessions."
        )
    st.bar_chart(
        summaries.assign(mean_percent=summaries["mean"] * 100),
        x="model_name",
        y="mean_percent",
        color="metric",
        stack=False,
        x_label="Model",
        y_label="Mean rate (%)",
    )
    st.dataframe(summaries, hide_index=True, width="stretch")
    st.caption(
        "Summary mean/std are rates (0-1); std is session variation, not a confidence interval."
    )


def live_scoring_demo():
    # This warning must precede every demo control and remain visible on reruns.
    st.warning(DEMO_WARNING)
    protocol = st.selectbox(
        "Demo protocol", ["dns", "icmp"], format_func=str.upper, key="scoring_protocol"
    )
    upload = st.file_uploader(
        "Upload an authorized lab capture",
        type=["pcap", "pcapng"],
        max_upload_size=50,
        help="Maximum 50 MiB. Capture bytes are sent to the configured backend for processing.",
    )
    retry = st.button("Retry scoring", disabled=upload is None)
    if upload is None:
        st.session_state.pop("demo_result", None)
        st.info("Upload a capture to score its flow windows with the final demo models.")
        return

    # Session-local results prevent duplicate uploads on unrelated widget reruns.
    # Do not globally cache captures or save uploaded traffic to the repository.
    key = (API_BASE_URL, protocol, hashlib.sha256(upload.getbuffer()).hexdigest())
    saved = st.session_state.get("demo_result")
    if retry or saved is None or saved[0] != key:
        st.session_state.pop("demo_result", None)
        with st.spinner("Extracting flow windows and scoring with both demo models..."):
            rows = request_backend(
                "POST",
                "/score",
                params={"protocol": protocol},
                files={"file": (upload.name, upload.getvalue(), "application/octet-stream")},
            )
        if rows is None:
            return
        saved = (key, rows)
        st.session_state["demo_result"] = saved

    flows = pd.DataFrame(saved[1])
    if flows.empty:
        st.info("No usable flow windows were returned for this capture.")
        return
    flows.insert(
        0,
        "status",
        flows["flagged_by_either"].map({True: "FLAGGED (demo only)", False: "Not flagged"}),
    )
    st.caption(
        f"{len(flows)} flow windows; {int(flows['flagged_by_either'].sum())} flagged by either model. "
        "A flag is a demo anomaly indication, not proof of a covert channel."
    )
    st.dataframe(flows, hide_index=True, width="stretch")
    st.subheader("Per-flow demo anomaly scores")
    st.scatter_chart(
        flows,
        x="isolation_forest_score",
        y="autoencoder_reconstruction_error",
        color="status",
        x_label="Isolation Forest anomaly score",
        y_label="Autoencoder reconstruction error",
    )
    st.caption(
        "Higher scores mean more anomalous on each axis; the two scales differ. "
        "The flag uses either saved training-score threshold, not necessarily both."
    )


def main():
    st.set_page_config(page_title="covertlens", layout="wide")
    st.title("covertlens")
    evaluation, demo = st.tabs(["Evaluation Results", "Live Scoring Demo"])
    with evaluation:
        evaluation_results()
    with demo:
        live_scoring_demo()


if __name__ == "__main__":
    main()
