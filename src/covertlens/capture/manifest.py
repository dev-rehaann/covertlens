"""Build an inventory of captured traffic files.

Expected filenames use the format
``{protocol}_{tool_or_baseline}_{condition}_{timestamp}.pcap``, where
``condition`` is either ``legit`` or ``covert`` and timestamps use UTC
``YYYYMMDDTHHMMSSZ`` form.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "capture_manifest.csv"
COLUMNS = [
    "filename",
    "protocol",
    "source",
    "condition",
    "timestamp",
    "file_size_bytes",
    "packet_count",
]
FILENAME_PATTERN = re.compile(
    r"^(?P<protocol>[^_]+)_(?P<source>[^_]+)_"
    r"(?P<condition>legit|covert)_(?P<timestamp>\d{8}T\d{6}Z)\.pcap$"
)


def parse_capture_filename(filename: str) -> dict[str, str] | None:
    """Return labels parsed from a valid capture filename, or ``None``."""
    match = FILENAME_PATTERN.fullmatch(filename)
    return match.groupdict() if match else None


def build_manifest(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Scan ``raw_dir`` and return one manifest row per valid pcap filename."""
    rows = []
    for capture_path in sorted(raw_dir.glob("*.pcap")):
        labels = parse_capture_filename(capture_path.name)
        if labels:
            rows.append(
                {
                    "filename": capture_path.name,
                    **labels,
                    "file_size_bytes": capture_path.stat().st_size,
                    "packet_count": None,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def write_manifest(
    raw_dir: Path = RAW_DIR, output_path: Path = MANIFEST_PATH
) -> pd.DataFrame:
    """Build and write the capture manifest, returning the DataFrame."""
    manifest = build_manifest(raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Count captures by protocol and condition."""
    return (
        manifest.groupby(["protocol", "condition"], sort=True)
        .size()
        .reset_index(name="count")
    )


def main() -> None:
    """Write the default manifest and print its protocol/condition counts."""
    manifest = write_manifest()
    print(summarize_manifest(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
