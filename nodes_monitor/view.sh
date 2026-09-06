#!/usr/bin/env bash
# Print the IPs of the nodes currently in a given state — a single read-only
# query round with machine-consumable stdout. All plumbing (config loading,
# node probing, state classification) comes from utils/monitor.sh.
#
#   ./view.sh --idle                        all idle nodes (the default filter)
#   ./view.sh --used --num 2                first 2 used nodes (node-list order)
#   ./view.sh --num 2 --hostfile /tmp/hostfile
#                                           also write an MPI-style hostfile
#   ./view.sh --config FILE                 config file (default: .data override,
#                                           else the bundled template)
#
# States (see get_node_state in utils/monitor.sh): IDLE USED TRAIN BROKEN
# OFFLINE NO_GPU UNVERIFIED.
#
# stdout carries NOTHING but the result — one line of space-separated IPs,
# an empty line when nothing matches — so it is safe to consume directly:
#   FREE=$(./view.sh --idle)
# Every diagnostic (usage errors, hostfile warnings) goes to stderr.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing — runs BEFORE sourcing utils/monitor.sh so that usage
# errors and --help are instant and never touch the config or the network.
#
# Everything defined here is view_-prefixed: utils/monitor.sh defines its
# own unprefixed helpers (load_config, run_node, ...), and sourcing it below
# would redefine (clobber) any unprefixed function of ours that is only
# called afterwards.
# ---------------------------------------------------------------------------

VIEW_STATE="IDLE"   # requested state filter (uppercase)
VIEW_STATE_SET=0    # at most one state flag is allowed (IDLE is the default,
                    # so a second flag cannot be told apart from the first)
VIEW_NUM="all"      # --num cap (positive integer) or "all"
VIEW_HOSTFILE=""    # --hostfile target path (empty: none)
VIEW_CONFIG=""      # --config FILE; handed to CONFIG_FILE after sourcing
VIEW_TMP=""         # mktemp -d dir with per-node state files (EXIT trap cleans)
VIEW_PIDS=()        # fetch subshell pids in flight (EXIT trap kills strays)

view_usage() {
    cat <<'EOF'
usage: view.sh [--idle|--used|--train|--broken|--offline|--no_gpu|--unverified]
               [--num N|all] [--hostfile PATH] [--config FILE] [-h|--help]

Print the IPs of the nodes currently in a given state (default: --idle).
stdout carries ONLY the result: one line of space-separated IPs (an empty
line when nothing matches). All diagnostics go to stderr.

  --idle|--used|--train|--broken|--offline|--no_gpu|--unverified
                        state filter; at most one may be given
  --num N|all           keep only the first N matching nodes, in node-list
                        order (fewer matches -> return what exists, no error)
  --hostfile PATH       also write an MPI-style hostfile for the selected
                        nodes: a '#hostfile' header, then '<ip> slots=<gpus>'
                        lines; nodes without a GPU count are skipped with a
                        warning on stderr
  --config FILE         config file (default: .data override, else bundled
                        template — same resolution as utils/monitor.sh)
  -h, --help            show this help and exit

Examples:
  view.sh --idle                          all idle nodes
  view.sh --used --num 2                  first 2 used nodes
  view.sh --num 2 --hostfile /tmp/hostfile
EOF
}

# Usage error: message + full help on stderr, non-zero exit. Only ever
# reached from view_parse_args, i.e. before any config or network access.
view_arg_error() {
    echo "Error: $1" >&2
    view_usage >&2
    exit 1
}

view_parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --idle|--used|--train|--broken|--offline|--no_gpu|--unverified)
                if (( VIEW_STATE_SET )); then
                    view_arg_error "only one state filter may be given (got '$1' on top of another)"
                fi
                VIEW_STATE_SET=1
                VIEW_STATE="${1#--}"       # strip the leading '--' ...
                VIEW_STATE="${VIEW_STATE^^}"  # ... and uppercase (no_gpu -> NO_GPU)
                shift
                ;;
            --num)
                [[ -n "${2:-}" ]] || view_arg_error "--num requires a value (positive integer or 'all')"
                VIEW_NUM="$2"
                if [[ "${VIEW_NUM}" != "all" ]]; then
                    [[ "${VIEW_NUM}" =~ ^[0-9]+$ ]] \
                        || view_arg_error "--num must be a positive integer or 'all' (got '${VIEW_NUM}')"
                    # 10# forces base 10: a '08'-style value would otherwise be
                    # read as an (invalid) octal literal and abort the
                    # arithmetic with a cryptic error. Also normalizes '007'->7.
                    VIEW_NUM=$((10#${VIEW_NUM}))
                    (( VIEW_NUM > 0 )) || view_arg_error "--num must be a positive integer or 'all' (got 0)"
                fi
                shift 2
                ;;
            --hostfile)
                [[ -n "${2:-}" ]] || view_arg_error "--hostfile requires a path"
                VIEW_HOSTFILE="$2"
                shift 2
                ;;
            --config)
                [[ -n "${2:-}" ]] || view_arg_error "--config requires a file"
                VIEW_CONFIG="$2"
                shift 2
                ;;
            -h|--help)
                view_usage
                exit 0
                ;;
            *)
                view_arg_error "unknown option: $1"
                ;;
        esac
    done
}

view_parse_args "$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared plumbing: load_config/finalize_config/load_nodes, get_node_state,
# INTERVAL, probe-cache windows, ... Sourcing runs nothing (the BASH_SOURCE
# guard at the bottom) and does not change the caller's set flags. It also
# re-derives SCRIPT_DIR itself — to .../nodes_monitor/utils, NOT the
# directory view.sh lives in — so nothing below this line may use SCRIPT_DIR.
source "${SCRIPT_DIR}/utils/monitor.sh"

# utils/monitor.sh's top level initializes CONFIG_FILE=""; hand our --config
# over only now — load_config honors a pre-set CONFIG_FILE.
CONFIG_FILE="${VIEW_CONFIG}"

# EXIT trap: kill fetch subshells still in flight, then drop the per-node
# state files. On a normal exit every pid has already been waited on (the
# kills are no-ops); when an outside SIGTERM tears the shell down mid-round,
# the still-running subshells would otherwise be orphaned together with
# their ssh/nvidia-smi children. TERM to the subshell suffices — its
# timeout-wrapped grandchild is bounded by RUN_NODE_TIMEOUT regardless.
# 2>/dev/null + || true because a cleanup failure must never mask the real
# exit status.
view_cleanup() {
    for p in ${VIEW_PIDS[@]+"${VIEW_PIDS[@]}"}; do
        kill -TERM "${p}" 2>/dev/null || true
    done
    [[ -n "${VIEW_TMP}" ]] && rm -rf "${VIEW_TMP}" 2>/dev/null || true
}

# Fetch one node's state and print it as a single parseable
# 'total|compute|state' line. Runs inside a background subshell so all nodes
# are probed in parallel (the old collect_stats pattern). '|| true' both tolerates a non-zero return from get_node_state and disables
# set -e inside it, so the nameref outputs are always fully populated.
view_fetch_one() {
    local ip="$1"
    local total=-1 compute=0 mem_used=0 trainer=0 state="UNVERIFIED"
    get_node_state "${ip}" total compute mem_used trainer state || true
    printf '%s|%s|%s\n' "${total}" "${compute}" "${state}"
}

# One parallel state-fetch round; each subshell writes '<tmp>/<idx>' via a
# .tmp + mv so a present file always holds a complete line — the read side
# never sees a half-written file, and a missing file means the fetch died.
view_collect_states() {
    VIEW_PIDS=()
    local i p
    for i in "${!IPS[@]}"; do
        ( view_fetch_one "${IPS[$i]}" > "${VIEW_TMP}/${i}.tmp" && mv "${VIEW_TMP}/${i}.tmp" "${VIEW_TMP}/${i}" ) &
        VIEW_PIDS+=($!)
    done
    for p in "${VIEW_PIDS[@]}"; do wait "${p}" 2>/dev/null || true; done
    # All fetched: empty the list so the EXIT trap has nothing to kill (a
    # waited pid could in principle be recycled between here and the trap).
    VIEW_PIDS=()
}

view_main() {
    load_config
    finalize_config
    load_nodes

    VIEW_TMP="$(mktemp -d /tmp/utils_view.XXXXXX)"
    trap view_cleanup EXIT

    view_collect_states

    # Keep the nodes matching the requested state, in node-list order, capped
    # at --num. A node whose state file never landed (or holds an empty state)
    # is UNVERIFIED: its fetch subshell died, so we genuinely do not know.
    local -a match_ips=() match_totals=()
    local i total compute state
    for i in "${!IPS[@]}"; do
        total=-1; compute=0; state="UNVERIFIED"
        # The -r guard matters: redirecting from a missing file would make
        # bash print its own 'No such file or directory' error on stderr.
        if [[ -r "${VIEW_TMP}/${i}" ]]; then
            IFS='|' read -r total compute state < "${VIEW_TMP}/${i}" || true
        fi
        [[ -n "${state}" ]] || state="UNVERIFIED"
        [[ "${state}" == "${VIEW_STATE}" ]] || continue
        match_ips+=("${IPS[$i]}")
        match_totals+=("${total}")
        if [[ "${VIEW_NUM}" != "all" ]] && (( ${#match_ips[@]} >= VIEW_NUM )); then
            break
        fi
    done

    if [[ -n "${VIEW_HOSTFILE}" ]]; then
        # slots=<gpus> needs a real count; without one the node is useless to
        # a launcher — warn and skip it in the FILE only (stdout still lists
        # the full selection). The C-style loop is set -u safe even with zero
        # matches. A bad path fails the redirect loudly under set -e.
        {
            printf '#hostfile\n'
            for (( i = 0; i < ${#match_ips[@]}; i++ )); do
                if (( match_totals[i] <= 0 )); then
                    echo "Warning: ${match_ips[$i]} has no GPU count (total=${match_totals[$i]}), skipped in hostfile" >&2
                    continue
                fi
                printf '%s slots=%s\n' "${match_ips[$i]}" "${match_totals[$i]}"
            done
        } > "${VIEW_HOSTFILE}"
    fi

    # The ONLY stdout output of a query run: one line of space-separated IPs
    # (an empty line when nothing matched). The [@]+ guard keeps the empty
    # array legal under set -u on bash < 4.4.
    local line=""
    local ip
    for ip in ${match_ips[@]+"${match_ips[@]}"}; do
        line+="${ip} "
    done
    printf '%s\n' "${line% }"
}

view_main
