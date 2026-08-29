"""Audit the flow-feature dataset before model training."""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd


FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv"
MIN_FLOWS = 30
PROTOCOLS = ("dns", "icmp")
KEY_FEATURES = [
    "packet_count",
    "duration_seconds",
    "size_mean",
    "size_std",
    "size_cv",
    "interarrival_mean",
    "interarrival_std",
    "interarrival_cv",
    "entropy_mean",
    "entropy_max",
    "is_single_packet_flow",
    "mean_query_length",
    "txt_null_ratio",
    "max_query_length",
    "icmp_size_cv",
]


def audit_dataset(features: pd.DataFrame) -> None:
    """Print sample sufficiency, missing-data, imbalance, and correlation checks."""
    required = ["protocol", "label", *KEY_FEATURES]
    missing = [column for column in required if column not in features]
    if missing:
        raise SystemExit(f"Feature dataset is missing columns: {', '.join(missing)}")

    features = features.copy()
    features["protocol"] = features["protocol"].astype(str).str.lower()
    features["label"] = pd.to_numeric(features["label"], errors="coerce")
    total = len(features)

    index = pd.MultiIndex.from_product([PROTOCOLS, (0, 1)], names=["protocol", "label"])
    counts = features.groupby(["protocol", "label"]).size().reindex(index, fill_value=0)
    count_table = counts.rename("count").reset_index()
    count_table["percentage"] = count_table["count"].div(total).mul(100) if total else 0.0

    print("DATASET SIZE")
    print(f"Total flows: {total}")
    print(
        count_table.to_string(
            index=False,
            formatters={"percentage": lambda value: f"{value:.2f}%"},
        )
    )

    short = count_table[count_table["count"] < MIN_FLOWS]
    if not short.empty:
        print("\n" + "!" * 72)
        print(f"WARNING: {len(short)} protocol/label group(s) have fewer than {MIN_FLOWS} flows")
        for row in short.itertuples(index=False):
            print(f"- {row.protocol}, label={row.label}: {row.count} flows")
        print("!" * 72)

    print("\nMISSING VALUES")
    nan_counts = features.isna().sum()
    nan_table = pd.DataFrame(
        {
            "nan_count": nan_counts,
            "nan_percentage": nan_counts.div(total).mul(100) if total else 0.0,
        }
    )
    print(
        nan_table.to_string(
            formatters={"nan_percentage": lambda value: f"{value:.2f}%"},
        )
    )
    dropped = total - len(features.dropna())
    dropped_percentage = dropped / total * 100 if total else 0.0
    print(f"Naive dropna() would drop {dropped}/{total} rows ({dropped_percentage:.2f}%).")

    print("\nCLASS IMBALANCE (legit:covert)")
    for name, frame in [("overall", features), *features.groupby("protocol")]:
        label_counts = frame["label"].value_counts()
        legit = int(label_counts.get(0, 0))
        covert = int(label_counts.get(1, 0))
        ratio = f"{legit / covert:.2f}:1" if covert else ("∞:1" if legit else "n/a")
        print(f"{name}: legit={legit}, covert={covert}, ratio={ratio}")

    print("\nFEATURE CORRELATION MATRIX (Pearson)")
    numeric_features = features[KEY_FEATURES].apply(pd.to_numeric, errors="coerce")
    correlation = numeric_features.corr()
    print(correlation.round(3).to_string())

    near_perfect = []
    for left, right in combinations(KEY_FEATURES, 2):
        value = correlation.at[left, right]
        if pd.notna(value) and abs(value) > 0.95:
            near_perfect.append((left, right, value))
    print("\nNEAR-PERFECT CORRELATIONS (|corr| > 0.95)")
    if near_perfect:
        for left, right, value in near_perfect:
            print(f"- {left} ↔ {right}: {value:.3f}")
    else:
        print("(none)")

    print("\nRECOMMENDATION")
    if short.empty:
        print("SUFFICIENT FOR SUPERVISED COMPARISON")
    else:
        print(
            "UNSUPERVISED-ONLY RECOMMENDED — insufficient covert samples for a supervised model"
        )
        details = ", ".join(
            f"{row.protocol}/label={row.label}: {row.count}"
            for row in short.itertuples(index=False)
        )
        print(f"Below 30 flows: {details}")


def main() -> None:
    """Load the repository dataset and print its audit."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not FEATURES_PATH.is_file():
        raise SystemExit(f"Feature dataset not found: {FEATURES_PATH}")
    audit_dataset(pd.read_csv(FEATURES_PATH))


if __name__ == "__main__":
    main()
