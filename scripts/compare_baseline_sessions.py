"""Compare DNS baselines without treating flow duration as capture duration.

Feature rows omit idle gaps and absolute packet timestamps, so their durations
cannot recover a session's elapsed time. Read that separately with capinfos
when the original pcap is available; do not substitute summed flow durations.
"""

import csv
import io
import shutil
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
RAW_DIR = REPO_ROOT / "data" / "raw"
FEATURES = [
    "size_cv",
    "interarrival_cv",
    "entropy_mean",
    "compression_ratio_mean",
    "mean_query_length",
    "duration_seconds",
    "packet_count",
]


def compare_baseline_sessions(features: pd.DataFrame, raw_dir: Path = RAW_DIR) -> None:
    """Print per-flow mean/sample std and separate raw-capture totals by session."""
    missing = sorted({"protocol", "label", "source_file", *FEATURES} - set(features.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    baseline = features.loc[features["protocol"].str.lower().eq("dns") & features["label"].eq(0)]
    if baseline.empty:
        print("No DNS legitimate sessions found.")
        return

    grouped = baseline.groupby("source_file", sort=True)
    print("DNS LEGIT BASELINES: per-flow mean and sample standard deviation")
    print(grouped[FEATURES].agg(["mean", "std"]).T.to_string(float_format="%.6f"))

    capinfos = shutil.which("capinfos")
    captures = []
    for source_file, rows in grouped:
        capture_path = raw_dir / Path(source_file).name
        packet_count = None
        duration_seconds = None
        if capinfos and capture_path.is_file():
            try:
                result = subprocess.run(
                    [capinfos, "-TmM", "-c", "-u", str(capture_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                metadata = list(csv.reader(io.StringIO(result.stdout)))[1]
                packet_count = int(metadata[1])
                duration_seconds = float(metadata[2])
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
                print(f"WARNING: raw capture metadata unavailable for {source_file}: {error}")
        else:
            print(f"WARNING: capinfos or original pcap unavailable for {source_file}.")
        captures.append(
            {
                "source_file": source_file,
                "flow_count": len(rows),
                "feature_packet_count_sum": int(rows["packet_count"].sum()),
                "raw_packet_count": packet_count,
                "raw_duration_seconds": duration_seconds,
                "raw_packets_per_second": (
                    packet_count / duration_seconds if duration_seconds else None
                ),
            }
        )
    print("\nRAW SESSION TOTALS: elapsed time from first to last captured packet")
    print(pd.DataFrame(captures).to_string(index=False, float_format="%.6f"))
    print("Flow duration describes a conversation/window, not the ten-minute capture.")
    print("These descriptive differences do not establish why a model flagged a flow.")


if __name__ == "__main__":
    compare_baseline_sessions(pd.read_csv(FEATURES_PATH))
