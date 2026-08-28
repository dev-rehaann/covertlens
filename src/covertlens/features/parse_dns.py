"""Extract packet-level DNS metadata from pcap files with PyShark."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyshark


logger = logging.getLogger("covertlens.features.parse_dns")

COLUMNS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "query_name",
    "query_length",
    "qtype",
    "answer_count",
    "payload_bytes",
    "packet_size",
]

# TXT (16) and NULL (10) records are common DNS tunneling carriers.
QTYPE_NAMES = {
    1: "A",
    2: "NS",
    5: "CNAME",
    6: "SOA",
    10: "NULL",
    12: "PTR",
    15: "MX",
    16: "TXT",
    28: "AAAA",
    33: "SRV",
    41: "OPT",
    255: "ANY",
}


def _field(layer: Any, name: str) -> Any | None:
    try:
        return getattr(layer, name)
    except (AttributeError, KeyError):
        try:
            return layer.get_field_value(name)
        except (AttributeError, KeyError):
            return None


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"0x[0-9a-f]+|\d+", str(value), flags=re.IGNORECASE)
    if not match:
        return None
    token = match.group()
    return int(token, 16) if token.lower().startswith("0x") else int(token)


def _qtype_name(value: Any) -> str | None:
    if value is None:
        return None
    label = re.split(r"[\s(]", str(value).strip().upper(), maxsplit=1)[0]
    if label in QTYPE_NAMES.values():
        return label
    code = _integer(value)
    return QTYPE_NAMES.get(code, f"TYPE{code}") if code is not None else label or None


def _hex_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    compact = str(value).replace(":", "").replace(" ", "")
    try:
        return bytes.fromhex(compact)
    except ValueError:
        return None


def _payload_bytes(packet: Any, dns: Any, query_name: str | None) -> bytes:
    for layer_name in ("udp", "tcp"):
        layer = getattr(packet, layer_name, None)
        payload = _hex_bytes(_field(layer, "payload")) if layer is not None else None
        if payload:
            if layer_name == "tcp" and len(payload) >= 2:
                declared_length = int.from_bytes(payload[:2], "big")
                if declared_length == len(payload) - 2:
                    payload = payload[2:]
            return payload

    # PyShark/TShark does not expose transport payload bytes consistently across
    # versions and export modes. Preserve observable DNS strings when raw bytes
    # are unavailable so later entropy code still has a documented approximation.
    text_parts = [value for value in (query_name, _field(dns, "txt")) if value]
    return "|".join(map(str, text_parts)).encode("utf-8", errors="replace")


def _packet_row(packet: Any) -> dict[str, Any]:
    dns = packet.dns
    ip = getattr(packet, "ip", None) or getattr(packet, "ipv6", None)
    transport = getattr(packet, "udp", None) or getattr(packet, "tcp", None)
    if ip is None or transport is None:
        raise ValueError("DNS packet has no supported IP/transport layer")

    query_name_value = _field(dns, "qry_name")
    query_name = str(query_name_value) if query_name_value else None
    is_response = _integer(_field(dns, "flags_response")) == 1

    return {
        "timestamp": float(packet.sniff_timestamp),
        "src_ip": str(ip.src),
        "dst_ip": str(ip.dst),
        "src_port": int(transport.srcport),
        "dst_port": int(transport.dstport),
        "query_name": query_name,
        "query_length": len(query_name) if query_name else 0,
        "qtype": _qtype_name(_field(dns, "qry_type")),
        "answer_count": _integer(_field(dns, "count_answers")) if is_response else None,
        "payload_bytes": _payload_bytes(packet, dns, query_name),
        "packet_size": int(packet.length),
    }


def extract_dns_packet_metadata(pcap_path: str) -> pd.DataFrame:
    """Return one metadata row per DNS packet in ``pcap_path``."""
    if not Path(pcap_path).is_file():
        raise FileNotFoundError(pcap_path)

    capture = pyshark.FileCapture(
        pcap_path,
        display_filter="dns",
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
                    "Skipping malformed DNS packet %s in %s: %s",
                    getattr(packet, "number", "unknown"),
                    pcap_path,
                    error,
                )
    except Exception as error:  # noqa: BLE001 - return rows parsed before a TShark failure.
        logger.error(
            "Stopped reading DNS capture %s after %d rows: %s",
            pcap_path,
            len(rows),
            error,
            exc_info=True,
        )
    finally:
        capture.close()

    return pd.DataFrame(rows, columns=COLUMNS)
