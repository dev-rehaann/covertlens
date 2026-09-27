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

# Presentation only: native Streamlit controls retain their keyboard/focus behavior.
STYLE = """
<style>
  .stMainBlockContainer { max-width: 1280px; padding: 2.25rem 3rem 3rem; }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stAppDeployButton"] { display: none; }
  .masthead { display: flex; justify-content: space-between; align-items: end;
      gap: 1.5rem; border-bottom: 1px solid color-mix(in srgb, currentColor 20%, transparent);
      padding-bottom: 1.5rem; }
  .masthead h1 { font-family: Georgia, serif; font-size: 2.7rem; font-weight: 400;
      letter-spacing: -.08rem; margin: 0; padding: 0; line-height: 1.1; }
  .masthead p { margin: .5rem 0 0; font-size: .95rem; }
  .masthead .edition { text-align: right; font-family: monospace; font-size: .75rem;
      line-height: 1.8; letter-spacing: .06em; white-space: nowrap; }
  [data-testid="stTabs"] [role="tablist"] { gap: 2rem; }
  [data-testid="stTabs"] [role="tab"] { padding: .9rem 0; height: auto; }
  [data-testid="stTabs"] [role="tab"] p { font-size: .95rem; font-weight: 600; }
  [data-testid="stTabs"] [role="tabpanel"] { padding-top: 1.5rem; }
  h2 { font-size: 1.6rem !important; font-weight: 600 !important; letter-spacing: -.025em; }
  h3 { font-size: 1.05rem !important; font-weight: 600 !important; }
  [data-testid="stAlert"] { border-radius: 3px; }
  [data-testid="stAlert"] p { font-size: .9rem; line-height: 1.6; }
  [data-testid="stButton"] button { border-radius: 4px; min-height: 44px; }
  [data-testid="stFileUploaderDropzone"] { border-radius: 4px; padding: 1.5rem; }
  [data-testid="stMetricValue"] { font-family: monospace; font-size: 2rem; }
  [data-testid="stMetricLabel"] { font-size: .85rem; }
  [data-testid="stDataFrame"] { border-radius: 3px; }
  [data-testid="stCaptionContainer"] { opacity: .82; }
  [data-testid="stCaptionContainer"] p { line-height: 1.6; }
  .research-footer { border-top: 1px solid color-mix(in srgb, currentColor 20%, transparent);
      padding-top: 1rem;
      margin-top: 2rem; font-family: monospace; font-size: .75rem; line-height: 1.8; }
  @media (max-width: 900px) {
    .stMainBlockContainer { padding: 1.5rem; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; }
  }
  @media (max-width: 600px) {
    .stMainBlockContainer { padding: 1.25rem 1rem; }
    .masthead { align-items: start; flex-direction: column; gap: .75rem; }
    .masthead .edition { text-align: left; }
    .masthead h1 { font-size: 2.2rem; }
    [data-testid="stTabs"] [role="tablist"] { gap: 1.25rem; }
  }
</style>
"""


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
    chart_text = "#c7d2cf" if st.context.theme.type == "dark" else "#43504b"
    title, controls = st.columns([3, 1], vertical_alignment="bottom")
    with title:
        st.header("Session-held-out evaluation")
        st.caption("Phase 3 / Leave-one-session-out / Real lab captures")
    with controls:
        protocol = st.selectbox(
            "Evaluation protocol", ["dns", "icmp"], format_func=str.upper, key="evaluation_protocol"
        )
    st.info(EVALUATION_CAPTION)
    st.button("Refresh saved results")
    with st.spinner("Loading saved LOSO results..."):
        results = request_backend("GET", f"/results/{protocol}")
    if results is None:
        return

    summaries = pd.concat(
        [
            pd.DataFrame(results["fpr_summary"]).assign(metric="FPR"),
            pd.DataFrame(results["recall_summary"]).assign(metric="Recall"),
        ],
        ignore_index=True,
    )
    if summaries.empty:
        st.info("No FPR/recall summaries are available for this run.")
        return
    # Pivot presentation only; never pool FPR and recall or average different models.
    summary = summaries.pivot(index="model_name", columns="metric", values=["mean", "count"])
    summary = summary.reindex(
        columns=pd.MultiIndex.from_product([["mean", "count"], ["FPR", "Recall"]])
    )
    summary = summary.reindex(["Isolation Forest", "Autoencoder", "XGBoost (supervised reference)"])
    summary = summary.dropna(how="all")
    summary["count"] = summary["count"].fillna(0)
    comparison, sessions = st.columns([1.5, 1], gap="large")
    with comparison:
        st.subheader("False alarms and detection")
        st.caption("Mean across held-out sessions. Lower FPR; higher recall.")
        chart = summaries.assign(mean_percent=summaries["mean"] * 100)
        chart["model"] = chart["model_name"].replace(
            {"XGBoost (supervised reference)": "XGBoost reference"}
        )
        st.vega_lite_chart(
            chart,
            {
                "height": 270,
                "mark": {"type": "bar", "cornerRadiusEnd": 2},
                "encoding": {
                    "y": {
                        "field": "model",
                        "type": "nominal",
                        "title": None,
                        "sort": ["Isolation Forest", "Autoencoder", "XGBoost reference"],
                    },
                    "yOffset": {"field": "metric", "sort": ["FPR", "Recall"]},
                    "x": {
                        "field": "mean_percent",
                        "type": "quantitative",
                        "title": "Mean rate (%)",
                        "scale": {"domain": [0, 100]},
                    },
                    "color": {
                        "field": "metric",
                        "type": "nominal",
                        "title": None,
                        "scale": {"domain": ["FPR", "Recall"], "range": ["#af703d", "#30796f"]},
                        "legend": {"orient": "top"},
                    },
                    "tooltip": [
                        {"field": "model_name", "title": "Model"},
                        {"field": "metric", "title": "Measure"},
                        {"field": "mean_percent", "title": "Mean (%)", "format": ".2f"},
                        {"field": "count", "title": "Held-out sessions"},
                    ],
                },
                "config": {
                    "view": {"stroke": None},
                    "axis": {
                        "labelFontSize": 12,
                        "labelColor": chart_text,
                        "titleColor": chart_text,
                    },
                    "legend": {"labelColor": chart_text},
                },
            },
            width="stretch",
            theme=None,
        )
    with sessions:
        st.subheader("Model comparison")
        st.caption("Unsupervised detectors; XGBoost is a supervised reference.")
        display = pd.DataFrame(
            {
                "Model": summary.index,
                "FPR": summary["mean"]["FPR"].map(
                    lambda value: f"{value:.2%}" if pd.notna(value) else "n/a"
                ),
                "Recall": summary["mean"]["Recall"].map(
                    lambda value: f"{value:.2%}" if pd.notna(value) else "n/a"
                ),
                "Legit sessions": summary["count"]["FPR"],
                "Covert sessions": summary["count"]["Recall"],
            }
        )
        st.dataframe(display, hide_index=True, width="stretch", row_height=48)
        for model_name, row in summary.iterrows():
            st.caption(
                f"{model_name}: n={int(row['count']['FPR'])} legit sessions; "
                f"n={int(row['count']['Recall'])} covert sessions."
            )
    st.caption(
        "Small session counts limit generalization. Flow windows are not independent sessions."
    )
    with st.expander("Session variation and metric definitions"):
        st.dataframe(summaries, hide_index=True, width="stretch")
        st.caption(
            "Mean/std are rates (0-1). Std measures session variation, not a confidence interval."
        )
    st.divider()
    st.subheader("Per-session results")
    st.caption(f"Saved run: {results.get('run_timestamp') or 'not recorded'}")
    folds = pd.DataFrame(results["per_fold_results"])
    preferred = ["fold", "model_name", "fold_type", "fpr", "recall", "tp", "fp", "tn", "fn"]
    st.dataframe(
        folds,
        hide_index=True,
        width="stretch",
        column_order=[name for name in preferred if name in folds]
        + [name for name in folds if name not in preferred],
        column_config={
            "fold": "Held-out capture",
            "model_name": "Model",
            "fold_type": "Session type",
            "fpr": st.column_config.NumberColumn("FPR", format="%.4f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
        },
    )


def live_scoring_demo():
    # This warning must precede every demo control and remain visible on reruns.
    st.warning(DEMO_WARNING)
    st.header("Capture analysis")
    st.caption("Phase 4 / Full-data models / Illustrative scoring only")
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
        with st.container(border=True):
            st.subheader("No capture selected")
            st.write("Choose a DNS or ICMP capture from your authorized lab.")
            st.caption("PCAP / PCAPNG / Up to 50 MiB. Extraction and scoring run on the local API.")
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
    # Windowed and unwindowed identifiers mix strings/integers; Arrow needs one type.
    flows["flow_id"] = flows["flow_id"].astype(str)
    flows.insert(
        0,
        "status",
        flows["flagged_by_either"].map({True: "Flagged - demo", False: "Not flagged"}),
    )
    flagged = int(flows["flagged_by_either"].sum())
    count, flags, packets = st.columns(3)
    count.metric("Flow windows", f"{len(flows):,}")
    flags.metric("Flagged by either model", f"{flagged:,}")
    packets.metric("Packets analyzed", f"{flows['packet_count'].sum():,}")
    st.caption("A flag is a demo anomaly indication, not proof of a covert channel.")
    st.subheader("Model scores by flow")
    chart_text = "#c7d2cf" if st.context.theme.type == "dark" else "#43504b"
    st.vega_lite_chart(
        flows,
        {
            "height": 320,
            "mark": {"type": "point", "filled": True, "size": 65, "opacity": 0.8},
            "encoding": {
                "x": {
                    "field": "isolation_forest_score",
                    "type": "quantitative",
                    "title": "Isolation Forest anomaly score",
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "autoencoder_reconstruction_error",
                    "type": "quantitative",
                    "title": "Autoencoder reconstruction error",
                },
                "color": {
                    "field": "status",
                    "type": "nominal",
                    "title": None,
                    "scale": {
                        "domain": ["Not flagged", "Flagged - demo"],
                        "range": ["#718096", "#af703d"],
                    },
                    "legend": {"orient": "top"},
                },
                "shape": {"field": "status", "type": "nominal", "legend": None},
                "tooltip": [
                    {"field": "flow_id", "title": "Flow"},
                    {"field": "status", "title": "Status"},
                    {"field": "isolation_forest_score", "format": ".4f"},
                    {"field": "autoencoder_reconstruction_error", "format": ".4f"},
                ],
            },
            "config": {
                "view": {"stroke": None},
                "axis": {"labelColor": chart_text, "titleColor": chart_text},
                "legend": {"labelColor": chart_text},
            },
        },
        width="stretch",
        theme=None,
    )
    st.caption(
        "Higher scores mean more anomalous on each axis; the two scales differ. "
        "The flag uses either saved training-score threshold, not necessarily both."
    )
    st.subheader("Flow inventory")
    st.dataframe(
        flows,
        hide_index=True,
        width="stretch",
        column_config={
            "flow_id": "Flow",
            "status": "Assessment",
            "packet_count": "Packets",
            "isolation_forest_score": st.column_config.NumberColumn(
                "Isolation Forest", format="%.5f"
            ),
            "autoencoder_reconstruction_error": st.column_config.NumberColumn(
                "Reconstruction error", format="%.5f"
            ),
        },
    )


def main():
    st.set_page_config(page_title="covertlens", layout="wide")
    st.html(STYLE)
    st.html("""<header class="masthead">
        <div><h1>covertlens<span style="color: #30796f">.</span></h1>
        <p>Network covert-channel research</p></div>
        <div class="edition">DNS / ICMP<br>RESEARCH WORKBENCH</div>
        </header>""")
    evaluation, demo = st.tabs(["Evaluation Results", "Live Scoring Demo"])
    with evaluation:
        evaluation_results()
    with demo:
        live_scoring_demo()
    st.html("""<footer class="research-footer">DEFENSIVE RESEARCH ONLY<br>
        Isolated lab captures. Session-held-out findings and full-data demo scores remain separate.
        </footer>""")


if __name__ == "__main__":
    main()
