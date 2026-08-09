#!/usr/bin/env bash
# Kill every process the GPU driver tracks (compute + graphics) on this node.
# Runs either on the host (remote nodes, via ssh) or inside the container
# (local node, via docker exec) — nvidia-smi reports namespace-local PIDs
# in both cases, so kill(1) works as-is.
#
# WARNING: on nodes with a GPU-attached display, --query-graphics-apps also
# lists the display server (Xorg/wayland) and it WILL be SIGKILL'd here —
# "everything nvidia-smi lists" is intentional.
set -euo pipefail

list_pids() {
    {
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true
        nvidia-smi --query-graphics-apps=pid --format=csv,noheader 2>/dev/null || true
    } | sort -un
}

pids="$(list_pids)"
if [[ -z "${pids}" ]]; then
    echo "no gpu compute processes"
    exit 0
fi

# Graceful SIGTERM first, then SIGKILL for survivors.
term_sent=0
while IFS= read -r pid; do
    if kill -TERM "${pid}" 2>/dev/null; then
        term_sent=$((term_sent + 1))
    fi
done <<< "${pids}"

sleep 3

killed=0
while IFS= read -r pid; do
    if kill -9 "${pid}" 2>/dev/null; then
        echo "killed ${pid}"
        killed=$((killed + 1))
    fi
done <<< "$(list_pids)"

# Processes that vanished after SIGTERM are the graceful kills; the SIGKILL
# count covers only the survivors.
echo "SIGTERM'd: ${term_sent}, SIGKILL'd: ${killed}"
