"""Print protocol-specific diagnostics for the latest flow-feature dataset.

Packet-size variance is intentionally treated as a two-sided signal. DNS
tunnels often fill queries to a consistent near-maximum size, producing lower
``size_cv`` than varied legitimate request/response traffic. ICMP tunnels can
vary payload sizes while ordinary OS ping traffic is nearly fixed-size,
producing higher ``size_cv``. Combining both protocols therefore hides useful
opposite-direction behavior.
"""

import sys
from pathlib import Path

import pandas as pd


FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv"
PROTOCOL_FEATURES = {
    "dns": [
        "size_cv",
        "interarrival_cv",
        "entropy_mean",
        "compression_ratio_mean",
        "mean_query_length",
        "txt_null_ratio",
    ],
    "icmp": [
        "size_cv",
        "interarrival_cv",
        "entropy_mean",
        "compression_ratio_mean",
        "icmp_size_cv",
    ],
}
TWO_SIDED_FEATURES = {"size_cv", "icmp_size_cv"}


def inspect_features(features: pd.DataFrame) -> None:
    """Print summaries and direction-aware sanity checks for each protocol."""
    for protocol, feature_names in PROTOCOL_FEATURES.items():
        protocol_rows = features.loc[features["protocol"].str.lower() == protocol]
        print(f"\n{protocol.upper()} FLOWS")
        for label, name in ((0, "legit"), (1, "covert")):
            print(f"\nLabel {label} ({name}) summary:")
            print(
                protocol_rows.loc[protocol_rows["label"] == label, feature_names]
                .describe()
                .to_string()
            )

        means = protocol_rows.groupby("label")[feature_names].mean().reindex([0, 1])
        print("\nSanity check:")
        for feature in feature_names:
            legit = means.at[0, feature]
            covert = means.at[1, feature]
            legit_text = "n/a" if pd.isna(legit) else f"{legit:.4f}"
            covert_text = "n/a" if pd.isna(covert) else f"{covert:.4f}"
            direction = "n/a" if pd.isna(legit) or pd.isna(covert) else (
                "↑" if covert > legit else "↓" if covert < legit else "→"
            )

            if feature in TWO_SIDED_FEATURES:
                print(
                    f"↕ {feature}: covert={covert_text}, legit={legit_text}, "
                    f"direction={direction} (two-sided expected)"
                )
            else:
                passes = pd.notna(legit) and pd.notna(covert) and covert > legit
                print(
                    f"{'✓' if passes else '✗'} {feature}: covert={covert_text}, "
                    f"legit={legit_text}, direction={direction} "
                    "(covert higher expected)"
                )


def main() -> None:
    """Show per-label statistics and whether covert means exceed legitimate means."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not FEATURES_PATH.is_file():
        raise SystemExit(f"Feature dataset not found: {FEATURES_PATH}")

    features = pd.read_csv(FEATURES_PATH)
    required = {"protocol", "label"}
    required.update(feature for names in PROTOCOL_FEATURES.values() for feature in names)
    missing = sorted(required.difference(features.columns))
    if missing:
        raise SystemExit(f"Feature dataset is missing columns: {', '.join(missing)}")
    inspect_features(features)


if __name__ == "__main__":
    main()
