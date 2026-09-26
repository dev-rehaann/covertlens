"""Train full-data dashboard demo models, not replacement LOSO evaluation models.

Both protocols use every available session, including legitimate and covert
flows, without passing labels to either trainer. The saved scaler and ordered
feature names must be reused at inference, not fitted again on an uploaded pcap.

The autoencoder file is a checkpoint containing state_dict, input_dim, and
demo-only provenance. Its anomaly_threshold and the Isolation Forest's saved
anomaly_threshold_ are calibrated on full-data training scores, never uploads.
Load the checkpoint with torch.load(..., weights_only=True), construct
FlowAutoencoder(input_dim), load_state_dict(checkpoint["state_dict"]), and eval().
Only load locally trusted joblib artifacts: joblib deserialization can execute
code. No held-out performance is computed or claimed by this module.
"""

import json
from pathlib import Path

import joblib
import torch

from covertlens.models.autoencoder_model import reconstruction_error, train_autoencoder
from covertlens.models.evaluate import training_score_threshold
from covertlens.models.isolation_forest_model import score_flows, train_isolation_forest
from covertlens.models.preprocess import load_and_prepare

REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
MODELS_PATH = REPO_ROOT / "models_release"
DEMO_WARNING = (
    "This model is trained on 100% of available data with nothing held out. "
    "It has NOT been evaluated with leave-one-session-out cross-validation "
    "like the models in Phase 3. Do not present its behavior as equivalent "
    "evidence to the LOSO results in docs/notes.md."
)


def train_final(features_csv_path: str, output_dir: str | Path = MODELS_PATH) -> None:
    """Train and save DNS/ICMP demo models and their full-data preprocessing.

    Ground-truth labels and session groups are intentionally unused. The JSON
    list defines the exact numeric feature order; metadata is not a model input.
    Re-running replaces these regeneratable artifacts, not the LOSO results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for protocol in ("dns", "icmp"):
        print(f"\nProtocol: {protocol}")
        print(f"WARNING: {DEMO_WARNING}")
        X, _, feature_names, scaler, _ = load_and_prepare(features_csv_path, protocol=protocol)
        isolation_forest = train_isolation_forest(X)
        autoencoder = train_autoencoder(X)
        # These full-data training thresholds serve the demo, not LOSO evaluation.
        isolation_forest.anomaly_threshold_ = training_score_threshold(
            score_flows(isolation_forest, X)
        )
        autoencoder_threshold = training_score_threshold(reconstruction_error(autoencoder, X))

        forest_path = output_dir / f"{protocol}_isolation_forest.joblib"
        autoencoder_path = output_dir / f"{protocol}_autoencoder.pt"
        scaler_path = output_dir / f"{protocol}_scaler.joblib"
        names_path = output_dir / f"{protocol}_feature_names.json"
        joblib.dump(isolation_forest, forest_path)
        torch.save(
            {
                "state_dict": autoencoder.state_dict(),
                "input_dim": len(feature_names),
                "training_flow_count": len(X),
                "demo_only": True,
                "warning": DEMO_WARNING,
                "anomaly_threshold": autoencoder_threshold,
            },
            autoencoder_path,
        )
        joblib.dump(scaler, scaler_path)
        names_path.write_text(json.dumps(feature_names, indent=2) + "\n", encoding="utf-8")

        print(f"Trained final Isolation Forest and Autoencoder on {len(X)} total {protocol} flows.")
        print(
            f"Features: {len(feature_names)}; all sessions included; labels not used for training."
        )
        print(
            "Full-training-score 90th-percentile thresholds: "
            f"Isolation Forest={isolation_forest.anomaly_threshold_:.6g}; "
            f"Autoencoder={autoencoder_threshold:.6g}"
        )
        for path in (forest_path, autoencoder_path, scaler_path, names_path):
            print(f"Saved: {path}")
        print(f"WARNING: {DEMO_WARNING}")


def main() -> None:
    """Train both protocols using the repository's local feature dataset."""
    train_final(str(FEATURES_PATH))


if __name__ == "__main__":
    main()
