#!/usr/bin/env bash
# SAFETY:
# THIS SCRIPT MUST ONLY BE RUN INSIDE THE ISOLATED LAB NETWORK DEFINED IN
# docs/lab-setup.md. DO NOT RUN ON HOST OR ANY PRODUCTION NETWORK.
# This script captures traffic only; it never installs, configures, or launches tunnel tools.
set -euo pipefail

TOOL="${1:-}"
DURATION="${2:-600}"

case "$TOOL" in
    iodine)
        CLIENT_COMMAND='iodine -f <server-ip> tunnel.example.lab'
        ;;
    dnscat2)
        CLIENT_COMMAND='dnscat --dns server=<server-ip>,domain=tunnel.example.lab'
        ;;
    *)
        echo "Usage: $0 {iodine|dnscat2} [duration_seconds]" >&2
        exit 2
        ;;
esac

if [[ ! "$DURATION" =~ ^[1-9][0-9]*$ ]]; then
    echo "Duration must be a positive integer number of seconds." >&2
    exit 2
fi

command -v tshark >/dev/null 2>&1 || {
    echo "Required command not found: tshark" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CAPTURE_DIR="$REPO_ROOT/data/raw"
CAPTURE_INTERFACE="${CAPTURE_INTERFACE:-any}"

mkdir -p "$CAPTURE_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CAPTURE_FILE="$CAPTURE_DIR/dns_${TOOL}_covert_${TIMESTAMP}.pcap"
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
CAPTURE_STARTED=$SECONDS

printf '\nMANUAL ACTION REQUIRED — isolated lab only.\n'
printf 'Use the command you manually verified in docs/lab-setup.md.\n'
printf 'Example only; run this yourself in another terminal:\n  %s\n\n' "$CLIENT_COMMAND"
read -r -p "Press enter once the tunnel is established and you've begun transferring test data..."

elapsed=$((SECONDS - CAPTURE_STARTED))
remaining=$((DURATION - elapsed))
((remaining < 0)) && remaining=0
echo "Capture will continue automatically for up to $remaining more seconds."

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
