"""Run and compare the Phase 3 unsupervised anomaly models."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from covertlens.models.autoencoder_model import reconstruction_error, train_autoencoder
from covertlens.models.evaluate import evaluate_scores
from covertlens.models.isolation_forest_model import score_flows, train_isolation_forest
from covertlens.models.preprocess import load_and_prepare


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
RESULTS_PATH = REPO_ROOT / "data" / "processed" / "model_comparison_results.csv"
DISPLAY_COLUMNS = ["model_name", "roc_auc", "avg_precision", "precision", "recall", "f1"]


def main() -> None:
    """Train both models and report held-out evaluation metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["dns", "icmp"])
    args = parser.parse_args()

    X, y, _, _ = load_and_prepare(str(FEATURES_PATH), protocol=args.protocol)
    X_train, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    isolation_forest = train_isolation_forest(X_train)
    isolation_result = evaluate_scores(
        y_test,
        score_flows(isolation_forest, X_test),
        "Isolation Forest",
    )

    # v1 simplification: train on all X_train regardless of evaluation label,
    # assuming covert contamination is rare; revisit with baseline-only training.
    autoencoder = train_autoencoder(X_train)
    autoencoder_result = evaluate_scores(
        y_test,
        reconstruction_error(autoencoder, X_test),
        "Autoencoder",
    )

    results = pd.DataFrame([isolation_result, autoencoder_result])
    print(
        results[DISPLAY_COLUMNS].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
