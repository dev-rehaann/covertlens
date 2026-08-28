"""Print quick diagnostics for the latest flow-feature dataset."""

import sys
from pathlib import Path

import pandas as pd


FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv"
KEY_FEATURES = [
    "size_cv",
    "interarrival_cv",
    "entropy_mean",
    "mean_query_length",
    "txt_null_ratio",
    "icmp_size_cv",
]


def main() -> None:
    """Show per-label statistics and whether covert means exceed legitimate means."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not FEATURES_PATH.is_file():
        raise SystemExit(f"Feature dataset not found: {FEATURES_PATH}")

    features = pd.read_csv(FEATURES_PATH)
    missing = [column for column in ["label", *KEY_FEATURES] if column not in features]
    if missing:
        raise SystemExit(f"Feature dataset is missing columns: {', '.join(missing)}")

    for label, name in ((0, "legit"), (1, "covert")):
        print(f"\nLabel {label} ({name}) summary:")
        print(features.loc[features["label"] == label, KEY_FEATURES].describe().to_string())

    means = features.groupby("label")[KEY_FEATURES].mean().reindex([0, 1])
    print("\nSanity check: covert mean > legit mean")
    for feature in KEY_FEATURES:
        legit = means.at[0, feature]
        covert = means.at[1, feature]
        passes = pd.notna(legit) and pd.notna(covert) and covert > legit
        legit_text = "n/a" if pd.isna(legit) else f"{legit:.4f}"
        covert_text = "n/a" if pd.isna(covert) else f"{covert:.4f}"
        print(f"{'✓' if passes else '✗'} {feature}: covert={covert_text}, legit={legit_text}")


if __name__ == "__main__":
    main()
