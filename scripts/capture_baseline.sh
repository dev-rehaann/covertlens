#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-600}"
if [[ ! "$DURATION" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [duration_seconds]" >&2
    exit 2
fi

for command in tshark getent; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "Required command not found: $command" >&2
        exit 1
    }
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DOMAIN_FILE="$SCRIPT_DIR/domains.txt"
CAPTURE_DIR="$REPO_ROOT/data/raw"
CAPTURE_INTERFACE="${CAPTURE_INTERFACE:-any}"

mapfile -t DOMAINS < <(grep -Ev '^[[:space:]]*(#|$)' "$DOMAIN_FILE")
if ((${#DOMAINS[@]} == 0)); then
    echo "No domains found in $DOMAIN_FILE" >&2
    exit 1
fi

mkdir -p "$CAPTURE_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CAPTURE_FILE="$CAPTURE_DIR/dns_baseline_legit_${TIMESTAMP}.pcap"
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

echo "Capturing DNS traffic on interface '$CAPTURE_INTERFACE' for $DURATION seconds..."
tshark -i "$CAPTURE_INTERFACE" -f "port 53" -a "duration:$DURATION" \
    -w "$CAPTURE_FILE" >/dev/null 2>&1 &
CAPTURE_PID=$!

index=$((RANDOM % ${#DOMAINS[@]}))
while kill -0 "$CAPTURE_PID" 2>/dev/null; do
    getent ahosts "${DOMAINS[$index]}" >/dev/null 2>&1 || true
    index=$(((index + 1) % ${#DOMAINS[@]}))
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
PACKET_COUNT="$(tshark -r "$CAPTURE_FILE" -T fields -e frame.number 2>/dev/null | wc -l)"

printf '\nCapture complete\n'
printf 'File: %s\n' "$CAPTURE_FILE"
printf 'Size: %s\n' "$FILE_SIZE"
printf 'Packets: %s\n' "$PACKET_COUNT"
