"""Extract packet-level ICMP metadata from pcap files with PyShark.

Real OS ping implementations use a FIXED default ICMP payload size (e.g. 56 bytes data on
Linux ping, 32 on Windows) — deviation from this fixed size across a flow's packets is itself
a strong tunneling signal, computed later in field_anomalies.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyshark

from covertlens.features.parse_dns import (
    _field,
    _hex_bytes,
    _integer,
    _nested_field,
    _timestamp,
)


logger = logging.getLogger("covertlens.features.parse_icmp")

COLUMNS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "icmp_type",
    "icmp_code",
    "payload_bytes",
    "payload_length",
    "packet_size",
    "ttl",
]


def _packet_row(packet: Any) -> dict[str, Any]:
    icmp = packet.icmp
    ip = getattr(packet, "ip", None)
    if ip is None:
        raise ValueError("ICMP packet has no IPv4 layer")

    payload = _hex_bytes(_nested_field(icmp, "icmp.data_raw"))
    data_layer = getattr(packet, "data", None)
    if payload is None and data_layer is not None:
        payload = _hex_bytes(_field(data_layer, "data"))
    payload = payload or b""

    return {
        "timestamp": _timestamp(packet),
        "src_ip": str(ip.src),
        "dst_ip": str(ip.dst),
        "icmp_type": _integer(_field(icmp, "type")),
        "icmp_code": _integer(_field(icmp, "code")),
        "payload_bytes": payload,
        "payload_length": len(payload),
        "packet_size": int(packet.length),
        # TTL anomalies are a known but secondary signal, retained for future analysis.
        "ttl": _integer(_field(ip, "ttl")),
    }


def extract_icmp_packet_metadata(pcap_path: str) -> pd.DataFrame:
    """Return one metadata row per ICMP packet in ``pcap_path``."""
    if not Path(pcap_path).is_file():
        raise FileNotFoundError(pcap_path)

    capture = pyshark.FileCapture(
        pcap_path,
        display_filter="icmp",
        keep_packets=False,
        include_raw=True,
        use_json=True,
    )
    rows = []
    try:
        for packet in capture:
            try:
                rows.append(_packet_row(packet))
            except Exception as error:  # noqa: BLE001 - malformed packets are isolated here.
                logger.warning(
                    "Skipping malformed ICMP packet %s in %s: %s",
                    getattr(packet, "number", "unknown"),
                    pcap_path,
                    error,
                )
    except Exception as error:  # noqa: BLE001 - return rows parsed before a TShark failure.
        logger.error(
            "Stopped reading ICMP capture %s after %d rows: %s",
            pcap_path,
            len(rows),
            error,
            exc_info=True,
        )
    finally:
        capture.close()

    return pd.DataFrame(rows, columns=COLUMNS)
