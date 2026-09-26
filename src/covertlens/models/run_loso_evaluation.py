"""Primary evaluation: leave one capture session out, grouped by source_file.

With three sessions per protocol this is only three-fold LOSO. Fold-to-fold
variation can reveal sensitivity to session-specific conditions and must be
reported; three sessions remain too thin for broad generalization claims.
Session grouping prevents leakage from correlated windowed sub-samples. Our
capture sessions each contain one class, making every holdout single-class:
ROC-AUC is undefined and AP cannot measure class discrimination in that fold.
For this novelty-detection regime, report FPR on legitimate holdouts and recall
on covert holdouts, separately. Averaging those rates together is meaningless.
Ranking metrics remain available in the saved results for future mixed-class
sessions; those sessions are excluded from the single-class rate summaries.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from covertlens.models.autoencoder_model import reconstruction_error, train_autoencoder
from covertlens.models.evaluate import evaluate_scores, evaluate_scores_no_threshold
from covertlens.models.isolation_forest_model import score_flows, train_isolation_forest
from covertlens.models.preprocess import load_and_prepare


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
RESULTS_PATH = REPO_ROOT / "data" / "processed" / "loso_results.csv"


def main() -> None:
    """Train on other sessions; calibrate scaling and thresholds on train only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["dns", "icmp"], required=True)
    parser.add_argument("--contamination", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.contamination <= 0.5:
        parser.error("--contamination must be in (0, 0.5]")

    X, y, names, preparation_scaler, groups = load_and_prepare(
        str(FEATURES_PATH), protocol=args.protocol
    )
    if groups.nunique() < 2:
        raise ValueError("LOSO requires at least two distinct capture sessions")
    # Undo convenience scaling, preserving deterministic imputation; only the
    # per-fold training rows determine the scaler actually used by the models.
    X = pd.DataFrame(preparation_scaler.inverse_transform(X), columns=names, index=X.index)
    print(f"Protocol: {args.protocol}; sessions: {groups.nunique()}; rows: {len(X)}")
    print(f"Training-score threshold percentile: {100 * (1 - args.contamination):.1f}")
    print("V1 uses all training rows without labels; mostly-normal data is not guaranteed.")
    results = []
    run_timestamp = datetime.now(timezone.utc).isoformat()

    for fold_number, (train, test) in enumerate(
        LeaveOneGroupOut().split(X, y, groups), start=1
    ):
        session = groups.iloc[test].iloc[0]
        y_test = y.iloc[test]
        train_legit = int(y.iloc[train].eq(0).sum())
        train_covert = int(y.iloc[train].eq(1).sum())
        test_legit = int(y_test.eq(0).sum())
        test_covert = int(y_test.eq(1).sum())
        training_had_no_legit_examples = train_legit == 0
        if test_covert == 0:
            fold_type = "legit-holdout"
        elif test_legit == 0:
            fold_type = "covert-holdout"
        else:
            fold_type = "mixed-holdout"
        print(f"\nFold {fold_number}: {session}")
        print(f"Fold type: {fold_type}")
        print(
            f"Train legit/covert: {train_legit}/{train_covert}; "
            f"test legit/covert: {test_legit}/{test_covert}"
        )
        if training_had_no_legit_examples:
            print("WARNING: ZERO legitimate examples in training; normality assumption fails.")

        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X.iloc[train]), columns=names, index=X.iloc[train].index
        )
        X_test = pd.DataFrame(
            scaler.transform(X.iloc[test]), columns=names, index=X.iloc[test].index
        )
        models = [
            ("Isolation Forest", train_isolation_forest, score_flows),
            ("Autoencoder", train_autoencoder, reconstruction_error),
        ]
        for model_name, trainer, scorer in models:
            if model_name == "Isolation Forest":
                model = trainer(X_train, contamination=args.contamination)
            else:
                model = trainer(X_train)
            training_scores = scorer(model, X_train)
            test_scores = scorer(model, X_test)
            metrics = evaluate_scores_no_threshold(y_test, test_scores, model_name)
            metrics.update(
                evaluate_scores(
                    y_test,
                    test_scores,
                    model_name,
                    training_scores=training_scores,
                    contamination=args.contamination,
                )
            )
            metrics["fpr"] = (
                metrics["fp"] / (metrics["fp"] + metrics["tn"])
                if fold_type == "legit-holdout" else float("nan")
            )
            metrics["recall"] = (
                metrics["tp"] / (metrics["tp"] + metrics["fn"])
                if fold_type == "covert-holdout" else float("nan")
            )
            results.append(
                {
                    "protocol": args.protocol,
                    "run_timestamp": run_timestamp,
                    "fold": session,
                    "fold_type": fold_type,
                    "training_had_no_legit_examples": training_had_no_legit_examples,
                    "train_legit": train_legit,
                    "train_covert": train_covert,
                    "test_legit": test_legit,
                    "test_covert": test_covert,
                    **metrics,
                }
            )

    table = pd.DataFrame(results)
    print("\nPER-FOLD RESULTS")
    print(
        table[
            [
                "fold", "fold_type", "model_name", "fpr", "recall",
                "tp", "fp", "tn", "fn", "training_had_no_legit_examples",
            ]
        ]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    for fold_type, metric in (("legit-holdout", "fpr"), ("covert-holdout", "recall")):
        print(f"\n{metric.upper()} SUMMARY: {fold_type} folds (sample std; count = sessions)")
        summary = table.loc[table["fold_type"] == fold_type].groupby("model_name")[metric]
        print(
            summary.agg(["mean", "std", "count"])
            .to_string(float_format=lambda value: f"{value:.4f}")
        )
    print("NaN rate means not applicable; std is unavailable with only one session.")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        table = pd.concat([pd.read_csv(RESULTS_PATH), table], ignore_index=True)
    table.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
