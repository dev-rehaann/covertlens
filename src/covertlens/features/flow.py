"""Group DNS and ICMP packets into short-lived bidirectional flows.

DNS requests/responses and ordinary ICMP exchanges are usually short lived. A
30-second inactivity timeout therefore comfortably separates unrelated
conversations, while a tunnel sending continuously should have short or
near-zero gaps and remain in one flow. The timeout is a tunable hyperparameter;
Phase 3 experiments may show that a different value separates sessions better.

Endpoint order is canonicalized internally so packets traveling in either
direction share a flow key. ICMP keys always use ``None`` for both ports.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class FlowKey:
    """Canonical identity for a bidirectional protocol conversation."""

    protocol: str
    src_ip: str
    dst_ip: str
    src_port_or_none: int | None
    dst_port_or_none: int | None


def _port(value: object) -> int | None:
    return None if value is None or value == "" else int(value)


def _flow_key(packet: dict[str, object]) -> FlowKey:
    protocol = str(packet["protocol"]).lower()
    src_port = None if protocol == "icmp" else _port(packet["src_port"])
    dst_port = None if protocol == "icmp" else _port(packet["dst_port"])
    src = (str(packet["src_ip"]), src_port)
    dst = (str(packet["dst_ip"]), dst_port)

    def endpoint_order(endpoint: tuple[str, int | None]) -> tuple[str, int]:
        return endpoint[0], -1 if endpoint[1] is None else endpoint[1]

    if endpoint_order(dst) < endpoint_order(src):
        src, dst = dst, src
    return FlowKey(protocol, src[0], dst[0], src[1], dst[1])


def _timestamp_seconds(value: object) -> float:
    timestamp = getattr(value, "timestamp", None)
    return float(timestamp()) if callable(timestamp) else float(str(value))


def assign_flow_id(
    packet_meta: list[dict[str, object]], timeout_seconds: float = 30
) -> pd.DataFrame:
    """Return packet metadata with inactivity-based integer ``flow_id`` values."""
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")

    state: dict[FlowKey, tuple[float, int]] = {}
    flow_ids: list[int] = []
    next_flow_id = 0

    for packet in packet_meta:
        key = _flow_key(packet)
        timestamp = _timestamp_seconds(packet["timestamp"])
        previous = state.get(key)

        if previous is None or timestamp - previous[0] > timeout_seconds:
            flow_id = next_flow_id
            next_flow_id += 1
        else:
            flow_id = previous[1]

        state[key] = timestamp, flow_id
        flow_ids.append(flow_id)

    frame = pd.DataFrame(packet_meta)
    frame["flow_id"] = pd.Series(flow_ids, dtype="int64")
    return frame
