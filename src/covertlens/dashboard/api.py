"""Local research API: static LOSO evidence and separately labeled live-demo scoring.

Run with python -m uvicorn covertlens.dashboard.api:app --host 127.0.0.1.
This unauthenticated lab demo is not a production upload service. Only locally
generated, trusted model artifacts are deserialized; uploads are packet captures,
never models. Temporary captures are removed on both success and failure.
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Response, UploadFile

from covertlens.features.aggregate import build_flow_features
from covertlens.features.flow import assign_flow_id
from covertlens.features.parse_dns import extract_dns_packet_metadata
from covertlens.features.parse_icmp import extract_icmp_packet_metadata
from covertlens.models.autoencoder_model import FlowAutoencoder, reconstruction_error
from covertlens.models.isolation_forest_model import score_flows
from covertlens.models.train_final import DEMO_WARNING

logger = logging.getLogger("covertlens.dashboard.api")
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = REPO_ROOT / "data" / "processed" / "loso_results.csv"
MODELS_PATH = REPO_ROOT / "models_release"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
Protocol = Literal["dns", "icmp"]
EXTRACTORS = {"dns": extract_dns_packet_metadata, "icmp": extract_icmp_packet_metadata}
app = FastAPI(title="covertlens", description=__doc__)


def _records(frame: pd.DataFrame) -> list[dict]:
    """Convert unavailable metrics (NaN) to JSON null, not invalid JSON numbers."""
    # Preserve score precision: to_json's decimal rounding can change flags
    # relative to near-zero Isolation Forest thresholds.
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "covertlens", "live_scoring_warning": DEMO_WARNING}


@app.get("/results/{protocol}", tags=["LOSO evaluation"])
def results(protocol: Protocol) -> dict:
    """Present the latest saved protocol run; never pool repeated historical folds."""
    if not RESULTS_PATH.is_file():
        raise HTTPException(
            404,
            "LOSO results not found. Run python -m covertlens.models.run_loso_evaluation "
            f"--protocol {protocol} first (run_loso_evaluation.py).",
        )
    try:
        table = pd.read_csv(RESULTS_PATH)
        table = table.loc[table["protocol"] == protocol].copy()
        if table.empty:
            raise HTTPException(
                404, f"No saved LOSO results for {protocol}; run run_loso_evaluation.py."
            )
        run_timestamp = None
        if "run_timestamp" in table:
            times = pd.to_datetime(table["run_timestamp"], utc=True, errors="raise")
            run_timestamp = str(table.loc[times.idxmax(), "run_timestamp"])
            table = table.loc[table["run_timestamp"] == run_timestamp]
        summaries = {}
        for fold_type, metric in (("legit-holdout", "fpr"), ("covert-holdout", "recall")):
            summary = (
                table.loc[table["fold_type"] == fold_type]
                .groupby("model_name")[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            summaries[f"{metric}_summary"] = _records(summary)
        return {
            "protocol": protocol,
            "evaluation": "leave-one-session-out",
            "run_timestamp": run_timestamp,
            "per_fold_results": _records(table),
            **summaries,
        }
    except HTTPException:
        raise
    except (ValueError, KeyError, OSError):
        logger.exception("Could not read saved LOSO results")
        raise HTTPException(500, "Saved LOSO results are unreadable; rerun run_loso_evaluation.py.")


def _load_models(protocol: Protocol):
    """Load only fixed local artifact paths; stale artifacts must be regenerated."""
    paths = [
        MODELS_PATH / f"{protocol}_{suffix}"
        for suffix in (
            "isolation_forest.joblib",
            "autoencoder.pt",
            "scaler.joblib",
            "feature_names.json",
        )
    ]
    if not all(path.is_file() for path in paths):
        raise HTTPException(
            400,
            "Final demo model files are missing. Run python -m "
            "covertlens.models.train_final first (train_final.py).",
        )
    try:
        forest = joblib.load(paths[0])
        checkpoint = torch.load(paths[1], map_location="cpu", weights_only=True)
        scaler = joblib.load(paths[2])
        names = json.loads(paths[3].read_text(encoding="utf-8"))
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) for name in names)
        ):
            raise ValueError("Invalid feature names")
        if (
            len(set(names)) != len(names)
            or list(scaler.feature_names_in_) != names
            or list(forest.feature_names_in_) != names
        ):
            raise ValueError("Feature order mismatch")
        if forest.n_features_in_ != len(names) or checkpoint["input_dim"] != len(names):
            raise ValueError("Model dimensions do not match features")
        thresholds = (float(forest.anomaly_threshold_), float(checkpoint["anomaly_threshold"]))
        if not np.isfinite(thresholds).all() or checkpoint["demo_only"] is not True:
            raise ValueError("Missing demo provenance or finite thresholds")
        autoencoder = FlowAutoencoder(checkpoint["input_dim"])
        autoencoder.load_state_dict(checkpoint["state_dict"])
        autoencoder.eval()
        return forest, autoencoder, scaler, names, thresholds
    except Exception:
        logger.exception("Final %s demo artifacts are invalid or outdated", protocol)
        raise HTTPException(
            400,
            "Final demo artifacts are invalid or lack saved thresholds. "
            "Rerun python -m covertlens.models.train_final (train_final.py).",
        )


@app.post("/score", tags=["Live demo - not LOSO evidence"], description=DEMO_WARNING)
def score(file: UploadFile, protocol: Protocol, response: Response) -> list[dict]:
    """Score uploaded flow windows against fixed full-data training thresholds.

    Flags use >=, matching evaluate.py's operating-threshold convention.
    build_flow_features directly reuses split_long_flow for 30-second windows.
    """
    # ponytail: synchronous local demo; add bounded jobs/auth before multi-user deployment.
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pcap", ".pcapng"}:
            raise HTTPException(400, "Upload a .pcap or .pcapng file.")
        forest, autoencoder, scaler, names, thresholds = _load_models(protocol)
        with tempfile.TemporaryDirectory(prefix="covertlens_upload_") as directory:
            capture_path = Path(directory) / f"capture{suffix}"
            size = 0
            with capture_path.open("wb") as target:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Capture exceeds the 50 MiB upload limit.")
                    target.write(chunk)
            with capture_path.open("rb") as capture:
                magic = capture.read(4)
            if magic not in {
                b"\xd4\xc3\xb2\xa1",
                b"\xa1\xb2\xc3\xd4",
                b"\x4d\x3c\xb2\xa1",
                b"\xa1\xb2\x3c\x4d",
                b"\x0a\x0d\x0d\x0a",
            }:
                raise HTTPException(400, "Malformed capture: no valid pcap/pcapng header.")
            try:
                # FastAPI sync routes run in worker threads. PyShark needs a
                # private subprocess-capable loop, especially on Windows.
                with asyncio.Runner():
                    packets = EXTRACTORS[protocol](str(capture_path))
                if packets.empty:
                    raise HTTPException(
                        400, f"Capture is malformed or contains no usable {protocol} packets."
                    )
                packets["protocol"] = protocol
                packets = assign_flow_id(packets.sort_values("timestamp").to_dict(orient="records"))
                features = build_flow_features(packets, protocol, include_timestamps=True)
                X = features.loc[:, names].astype(float)
                # Aggregation already supplies zero timing statistics for single
                # packets; protocol-irrelevant columns are excluded by saved names.
                if not np.isfinite(X.to_numpy()).all():
                    raise ValueError("Non-finite flow features")
            except HTTPException:
                raise
            except Exception:
                logger.exception("Could not extract usable %s flow features", protocol)
                raise HTTPException(
                    400,
                    "Could not parse usable flow features from this capture. "
                    "Check the capture format, protocol, and TShark installation.",
                )

            try:
                X = pd.DataFrame(scaler.transform(X), columns=names, index=features.index)
                forest_scores = score_flows(forest, X)
                autoencoder_scores = reconstruction_error(autoencoder, X)
                if not (np.isfinite(forest_scores).all() and np.isfinite(autoencoder_scores).all()):
                    raise ValueError("Non-finite anomaly scores")
            except (ValueError, RuntimeError, OverflowError):
                logger.exception("Could not score finite %s demo features", protocol)
                raise HTTPException(400, "Capture features or scores are invalid or out of range.")
            output = features[
                ["flow_id", "start_timestamp", "end_timestamp", "packet_count"]
            ].copy()
            output["isolation_forest_score"] = forest_scores
            output["autoencoder_reconstruction_error"] = autoencoder_scores
            output["flagged_by_either"] = forest_scores.ge(thresholds[0]) | autoencoder_scores.ge(
                thresholds[1]
            )
            response.headers["X-Covertlens-Evidence"] = "full-data-live-demo-not-LOSO"
            response.headers["X-Covertlens-Warning"] = DEMO_WARNING
            return _records(output)
    finally:
        file.file.close()
