#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-600}"
if [[ ! "$DURATION" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [duration_seconds]" >&2
    exit 2
fi

for command in tshark capinfos ping; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "Required command not found: $command" >&2
        exit 1
    }
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_FILE="$SCRIPT_DIR/ping_targets.txt"
CAPTURE_DIR="$REPO_ROOT/data/raw"
CAPTURE_INTERFACE="${CAPTURE_INTERFACE:-any}"

mapfile -t TARGETS < <(grep -Ev '^[[:space:]]*(#|$)' "$TARGET_FILE")
if ((${#TARGETS[@]} == 0)); then
    echo "No ping targets found in $TARGET_FILE" >&2
    exit 1
fi

mkdir -p "$CAPTURE_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CAPTURE_FILE="$CAPTURE_DIR/icmp_baseline_legit_${TIMESTAMP}.pcap"
CAPTURE_PID=""

stop_capture() {
    if [[ -n "$CAPTURE_PID" ]]; then
        if kill -0 "$CAPTURE_PID" 2>/dev/null; then
            kill -INT "$CAPTURE_PID" 2>/dev/null || true
        fi
        wait "$CAPTURE_PID" 2>/dev/null || true
        CAPTURE_PID=""
    fi
}

trap 'stop_capture; exit 130' INT TERM
trap stop_capture EXIT

echo "Capturing ICMP traffic on interface '$CAPTURE_INTERFACE' for $DURATION seconds..."
tshark -i "$CAPTURE_INTERFACE" -f "icmp" -a "duration:$DURATION" \
    -w "$CAPTURE_FILE" >/dev/null 2>&1 &
CAPTURE_PID=$!

index=$((RANDOM % ${#TARGETS[@]}))
varied_sizes=(48 64 72)
while kill -0 "$CAPTURE_PID" 2>/dev/null; do
    target="${TARGETS[$index]}"
    if ((RANDOM % 4 == 0)); then
        size="${varied_sizes[RANDOM % ${#varied_sizes[@]}]}"
        ping -c 1 -W 2 -s "$size" "$target" >/dev/null 2>&1 || true
    else
        ping -c 1 -W 2 "$target" >/dev/null 2>&1 || true
    fi
    index=$(((index + 1) % ${#TARGETS[@]}))
    sleep $((1 + RANDOM % 8))
done

if wait "$CAPTURE_PID"; then
    capture_status=0
else
    capture_status=$?
fi
CAPTURE_PID=""
trap - EXIT INT TERM
if ((capture_status != 0)); then
    echo "tshark exited with status $capture_status" >&2
    exit "$capture_status"
fi

FILE_SIZE="$(du -h "$CAPTURE_FILE" | awk '{print $1}')"
PACKET_COUNT="$(capinfos -c "$CAPTURE_FILE" | awk -F: '/Number of packets/ {gsub(/[[:space:]]/, "", $2); print $2}')"

printf '\nCapture complete\n'
printf 'File: %s\n' "$CAPTURE_FILE"
printf 'Size: %s\n' "$FILE_SIZE"
printf 'Packets: %s\n' "$PACKET_COUNT"
