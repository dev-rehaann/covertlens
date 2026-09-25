"""Time-window subdivision for long, continuously active flows."""

from __future__ import annotations

import pandas as pd


def split_long_flow(
    packet_df_for_one_flow: pd.DataFrame,
    window_seconds: float = 30.0,
    min_duration_to_split: float = 60.0,
) -> list[pd.DataFrame]:
    """Split a long flow into consecutive, non-overlapping time windows.

    Short flows are returned unchanged. Non-overlapping windows are deliberate:
    overlapping windows from one continuous session are autocorrelated and would
    inflate the apparent sample size without adding independent observations.
    Even non-overlapping windows from the same capture remain partly dependent
    because they share a tunnel tool, run, and network conditions; windowing
    improves granularity but does not replace independent capture sessions.

    A final one-packet window is merged into its predecessor so it does not
    become a near-empty feature sample.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if min_duration_to_split < 0:
        raise ValueError("min_duration_to_split must be non-negative")
    if packet_df_for_one_flow.empty:
        return []

    timestamps = packet_df_for_one_flow["timestamp"].astype(float)
    duration = float(timestamps.max() - timestamps.min())
    if duration < min_duration_to_split:
        return [packet_df_for_one_flow]

    packets = packet_df_for_one_flow.sort_values("timestamp")
    window_numbers = (
        packets["timestamp"].astype(float) - timestamps.min()
    ) // window_seconds
    windows = [window for _, window in packets.groupby(window_numbers, sort=True)]

    if len(windows) > 1 and len(windows[-1]) < 2:
        windows[-2] = pd.concat([windows[-2], windows[-1]])
        windows.pop()
    return windows
