"""Build the flow-level feature dataset from the capture manifest."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from covertlens.features.aggregate import COLUMNS as FLOW_FEATURE_COLUMNS
from covertlens.features.aggregate import build_flow_features
from covertlens.features.flow import assign_flow_id
from covertlens.features.parse_dns import extract_dns_packet_metadata
from covertlens.features.parse_icmp import extract_icmp_packet_metadata


logger = logging.getLogger("covertlens.features.build_dataset")

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "capture_manifest.csv"
RAW_DIR = REPO_ROOT / "data" / "raw"
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
LABELS = {"legit": 0, "covert": 1}
EXTRACTORS: dict[str, Callable[[str], pd.DataFrame]] = {
    "dns": extract_dns_packet_metadata,
    "icmp": extract_icmp_packet_metadata,
}


def build_dataset(
    manifest_path: Path = MANIFEST_PATH,
    raw_dir: Path = RAW_DIR,
    output_path: Path = FEATURES_PATH,
) -> pd.DataFrame:
    """Process every manifested pcap, write ``features.csv``, and return it."""
    manifest = pd.read_csv(manifest_path)
    tables = []
    failures: list[tuple[str, str]] = []
    total = len(manifest)

    for index, capture in enumerate(manifest.itertuples(index=False), start=1):
        filename = str(capture.filename)
        protocol = str(capture.protocol).lower()
        logger.info("Processing %s (%d of %d)", filename, index, total)

        try:
            pcap_path = Path(filename)
            if not pcap_path.is_absolute():
                pcap_path = raw_dir / pcap_path
            packets = EXTRACTORS[protocol](str(pcap_path))
            packets["protocol"] = protocol
            packets = packets.sort_values("timestamp")
            packets = assign_flow_id(packets.to_dict(orient="records"))
            features = build_flow_features(packets, protocol)
            features["label"] = LABELS[str(capture.condition).lower()]
            features["source_file"] = filename
            tables.append(features)
        except Exception as error:  # noqa: BLE001 - one bad capture must not stop the dataset.
            failures.append((filename, str(error)))
            logger.error("Failed to process %s: %s", filename, error, exc_info=True)

    columns = [*FLOW_FEATURE_COLUMNS, "label", "source_file"]
    dataset = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)
    dataset = dataset.reindex(columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"Total flows: {len(dataset)}")
    print("Flows by protocol and label:")
    counts = dataset.groupby(["protocol", "label"]).size().reset_index(name="count")
    print(counts.to_string(index=False) if not counts.empty else "(none)")
    print("Failed pcaps:")
    if failures:
        for filename, error in failures:
            print(f"- {filename}: {error}")
    else:
        print("(none)")

    return dataset


def main() -> None:
    """Build the dataset using repository-default paths."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_dataset()


if __name__ == "__main__":
    main()
