"""Exploratory, non-leak-free quick check using random row splits.

Use run_loso_evaluation as the primary evaluation path for the writeup:
this quick check shares capture sessions across training and test rows.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from covertlens.models.autoencoder_model import reconstruction_error, train_autoencoder
from covertlens.models.evaluate import evaluate_scores
from covertlens.models.isolation_forest_model import score_flows, train_isolation_forest
from covertlens.models.preprocess import load_and_prepare


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
RESULTS_PATH = REPO_ROOT / "data" / "processed" / "model_comparison_results.csv"
DISPLAY_COLUMNS = [
    "model_name", "roc_auc", "avg_precision", "precision", "recall", "f1",
    "tp", "fp", "tn", "fn",
]


def main() -> None:
    """Train both models and report held-out evaluation metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["dns", "icmp"])
    args = parser.parse_args()

    X, y, feature_names, preparation_scaler, _ = load_and_prepare(
        str(FEATURES_PATH), protocol=args.protocol
    )
    # Recover the imputed raw features so evaluation uses a scaler fitted only
    # on training rows; the convenience preprocessor fits on the whole input.
    X = pd.DataFrame(
        preparation_scaler.inverse_transform(X), columns=feature_names, index=X.index
    )
    X_train, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    isolation_forest = train_isolation_forest(X_train)
    isolation_result = evaluate_scores(
        y_test,
        score_flows(isolation_forest, X_test),
        "Isolation Forest",
        training_scores=score_flows(isolation_forest, X_train),
    )

    # v1: use the full training split without label filtering. This lab dataset
    # has a covert majority, so the mostly-normal assumption does not hold here.
    autoencoder = train_autoencoder(X_train)
    autoencoder_result = evaluate_scores(
        y_test,
        reconstruction_error(autoencoder, X_test),
        "Autoencoder",
        training_scores=reconstruction_error(autoencoder, X_train),
    )

    results = pd.DataFrame([isolation_result, autoencoder_result])
    results["protocol"] = args.protocol or "combined"
    results["run_timestamp"] = datetime.now(timezone.utc).isoformat()
    print(f"Protocol: {args.protocol or 'combined'}")
    print(
        results[DISPLAY_COLUMNS].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("Thresholds use the training-score 90th percentile; metrics remain exploratory.")
    print("Random row split can share capture sessions across train and test.")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        results = pd.concat([pd.read_csv(RESULTS_PATH), results], ignore_index=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
