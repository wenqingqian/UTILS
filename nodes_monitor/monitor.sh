#!/usr/bin/env bash
# Monitor GPU status across the nodes defined in config.toml.
#   ./monitor.sh                 : monitor only
#   ./monitor.sh --train         : monitor and auto-start training on idle nodes
#   ./monitor.sh --release       : kill all tagged trainer jobs and exit
#   ./monitor.sh --kill 8,42     : kill GPU-occupying processes on those node suffixes and exit
#   ./monitor.sh --exclude 0,1   : skip nodes by index (0-based position in the node list)
#   ./monitor.sh --config FILE   : config file (default: .data override, else bundled template)
#
# Config resolution order (highest first):
#   1. --config FILE
#   2. .data/nodes_monitor/config.toml   (gitignored, the real/debug config)
#   3. nodes_monitor/config.toml         (committed template, placeholder values)
#
# Requires python3 >= 3.11 (tomllib) on the machine running this script.
# NOTE: --exclude indices are 0-based here; manager.sh --exclude uses 1-based
# workbench indices (converted internally).

# NOTE: `set -euo pipefail` is applied at the bottom, inside the standalone
# guard, so that sourcing this file (manager.sh) does not implicitly change
# the caller's error handling (callers must set their own flags first).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE=""

# ---------------------------------------------------------------------------
# Sectioned TOML loading (embedded python tomllib -> KEY=value -> bash source)
# ---------------------------------------------------------------------------

# Parse the TOML into flat KEY=value lines (sections joined with '_', lists as
# newline-separated KEY+=item) on stdout. Requires python3 >= 3.11 (tomllib).
# Values are emitted raw (no quoting/escaping) so bash reads them verbatim;
# the only protocol limitation is that values must not contain newlines.
_toml_to_env() {
    local file="$1"
    python3 - "$file" <<'PY'
import sys, tomllib

def walk(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(f"{prefix}{k}_" if prefix else f"{k}_", v, out)
    elif isinstance(obj, list):
        for item in obj:
            out.append(f"{prefix[:-1]}+={item}")
    else:
        out.append(f"{prefix[:-1]}={obj}")

with open(sys.argv[1], "rb") as fh:
    data = tomllib.load(fh)
lines = []
walk("", data, lines)
print("\n".join(lines))
PY
}

# Pick the effective config file.
resolve_config() {
    if [[ -n "${CONFIG_FILE}" ]]; then
        [[ -r "${CONFIG_FILE}" ]] || { echo "Error: config not readable: ${CONFIG_FILE}" >&2; exit 1; }
        return
    fi
    local data_cfg="${UTILS_ROOT}/.data/nodes_monitor/config.toml"
    local tmpl_cfg="${SCRIPT_DIR}/config.toml"
    if [[ -r "${data_cfg}" ]]; then
        CONFIG_FILE="${data_cfg}"
    elif [[ -r "${tmpl_cfg}" ]]; then
        CONFIG_FILE="${tmpl_cfg}"
    else
        echo "Error: no config.toml found (looked in .data and template)" >&2
        exit 1
    fi
}

# ---- defaults (used when the config does not provide a value) ----
SSH_IDENTITY=""
SSH_PORT=2222
WORKSPACE_ROOT=""
LAUNCHER_HOST=""
CONTAINER=""
TRAINER_MARKER="__UTILS_train_job__"
# Derived from TRAINER_MARKER in finalize_config when not set in config.
PGREP_PATTERN=""
TRAINER_COMMAND=""
PROBE_COMMAND=""
THEME="table"
NODES_FILE=""
NODES_INLINE=()

IPS=()
IP_RE='^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}$'
SSH_OPTS=()

# Trainers launched by THIS session (via launch_trainer). cleanup() kills them
# on exit (INT/TERM/EXIT trap) so no process is ever left behind on the nodes
# — regardless of how the monitor ends. Tracked at the CALL SITES, not inside
# launch_trainer: callers background the launcher in a subshell, where array
# mutations would be lost.
LAUNCHED_IPS=()

# Node-side helper scripts (in-container paths, derived from workspace_root).
# They run inside the container environment on the local node (docker exec)
# and via plain ssh on remote nodes — which requires the workspace to be
# mounted at workspace_root on the remote hosts as well. The CUDA probe's
# interpreter is configurable via [trainer] probe_command (R3: the system
# python3 may not see torch on nodes where it lives in a conda env).
CUDA_PROBE=""
KILL_GPU_SCRIPT=""
LOG_FILE="/tmp/utils_train.log"

# Colors for terminal output.
C_RESET=$'\033[0m'
C_BOLD=$'\033[1m'
C_RED=$'\033[31m'
C_GREEN=$'\033[32m'
C_YELLOW=$'\033[33m'
C_BLUE=$'\033[34m'
C_CYAN=$'\033[36m'
C_GRAY=$'\033[90m'

# Tuning knobs; all overridable via the [monitor] section of the config.
INTERVAL=5
COOLDOWN=60
# CUDA health-probe cadence: one probe at startup, then every PROBE_INTERVAL
# for confirmed verdicts. Inconclusive probes expire after COOLDOWN instead.
PROBE_INTERVAL=600

COMPUTE_THRESHOLD=0
MEM_USED_THRESHOLD=100

LOCAL_IPS="$(hostname -I 2>/dev/null || hostname -i 2>/dev/null || true)"

MODE="monitor"
EXCLUDE_INDICES=()
KILL_NODE_SUFFIXES=()

load_config() {
    resolve_config
    local line toml_env
    # Fail loudly instead of silently keeping defaults when the parser itself
    # fails (python3 < 3.11 / missing python3 / TOML syntax error).
    if ! toml_env="$(_toml_to_env "${CONFIG_FILE}")"; then
        echo "Error: failed to parse config: ${CONFIG_FILE} (need python3 >= 3.11 with tomllib)" >&2
        exit 1
    fi
    while IFS= read -r line; do
        case "${line}" in
            nodes_list\+=*)   NODES_INLINE+=("${line#nodes_list+=}") ;;
            ssh_identity=*)   SSH_IDENTITY="${line#ssh_identity=}"   ;;
            ssh_port=*)       SSH_PORT="${line#ssh_port=}"           ;;
            paths_workspace_root=*) WORKSPACE_ROOT="${line#paths_workspace_root=}" ;;
            paths_launcher_host=*)  LAUNCHER_HOST="${line#paths_launcher_host=}"   ;;
            env_container=*)  CONTAINER="${line#env_container=}"     ;;
            trainer_marker=*) TRAINER_MARKER="${line#trainer_marker=}" ;;
            trainer_pgrep_pattern=*) PGREP_PATTERN="${line#trainer_pgrep_pattern=}" ;;
            trainer_command=*) TRAINER_COMMAND="${line#trainer_command=}" ;;
            trainer_probe_command=*) PROBE_COMMAND="${line#trainer_probe_command=}" ;;
            ui_theme=*)       THEME="${line#ui_theme=}"             ;;
            nodes_file=*)     NODES_FILE="${line#nodes_file=}"       ;;
            monitor_interval=*)  INTERVAL="${line#monitor_interval=}" ;;
            monitor_cooldown=*)  COOLDOWN="${line#monitor_cooldown=}" ;;
            monitor_probe_interval=*) PROBE_INTERVAL="${line#monitor_probe_interval=}" ;;
            monitor_compute_threshold=*) COMPUTE_THRESHOLD="${line#monitor_compute_threshold=}" ;;
            monitor_mem_used_threshold=*) MEM_USED_THRESHOLD="${line#monitor_mem_used_threshold=}" ;;
            monitor_log_file=*) LOG_FILE="${line#monitor_log_file=}" ;;
            *)
                # A typo like `pgrep_pattern` under [trainer] would silently
                # keep the default — warn so the operator notices.
                echo "Warning: ignoring unknown config key: ${line%%=*}" >&2
                ;;
        esac
    done <<< "${toml_env}"
}

# Called after load_config: derive dependent values, enforce sane defaults,
# and auto-fix the SSH key permissions (OpenSSH refuses keys readable by others).
finalize_config() {
    [[ -n "${WORKSPACE_ROOT}" ]] || { echo "Error: [paths] workspace_root is required" >&2; exit 1; }
    [[ -n "${SSH_IDENTITY}" ]]   || { echo "Error: [ssh] identity is required" >&2; exit 1; }
    [[ -n "${CONTAINER}" ]]      || { echo "Error: [env] container is required" >&2; exit 1; }

    # Keep the pgrep/pkill pattern in sync with the marker: derive it from the
    # marker when the config does not set it. The [X]first-char bracket trick
    # stops pgrep/pkill from matching their own command line.
    if [[ -z "${PGREP_PATTERN}" ]]; then
        PGREP_PATTERN="[${TRAINER_MARKER:0:1}]${TRAINER_MARKER:1}"
    fi

    # Launcher script as seen INSIDE the container (docker exec on the local
    # node; the remote branch runs the same in-container path over ssh).
    # Defaults are derived from workspace_root so all three helpers
    # (launcher, CUDA probe, gpu_kill) live in the same mount point.
    [[ -n "${LAUNCHER_HOST}" ]] || LAUNCHER_HOST="${WORKSPACE_ROOT}/UTILS/nodes_monitor/run_train.sh"

    CUDA_PROBE="${WORKSPACE_ROOT}/UTILS/nodes_monitor/cuda_probe.py"
    KILL_GPU_SCRIPT="${WORKSPACE_ROOT}/UTILS/nodes_monitor/gpu_kill.sh"

    # NOTE: StrictHostKeyChecking=no + UserKnownHostsFile=/dev/null disable
    # host-key verification — an accepted MITM exposure for internal cluster
    # tooling behind a jump host.
    #
    # ServerAliveInterval/ServerAliveCountMax detect a node dying mid-command
    # (~15s) so the strictly sequential monitor loop cannot stall on TCP
    # retransmission timeouts.
    SSH_OPTS=(
        -p "${SSH_PORT}"
        -i "${SSH_IDENTITY}"
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
        -o ConnectTimeout=5
        -o ServerAliveInterval=5
        -o ServerAliveCountMax=3
        -o BatchMode=yes
        -o LogLevel=ERROR
    )

    # Auto-fix SSH private key permissions: OpenSSH ignores keys that are
    # group/world-readable. Tighten to 600 if too open so auth does not fail.
    # (GNU-specific stat -c; the 8# base forces octal interpretation.)
    if [[ -f "${SSH_IDENTITY}" ]]; then
        local perm
        perm="$(stat -c '%a' "${SSH_IDENTITY}" 2>/dev/null || echo "")"
        if [[ -n "${perm}" && $(( 8#${perm} & 077 )) -ne 0 ]]; then
            if ! chmod 600 "${SSH_IDENTITY}" 2>/dev/null; then
                echo "Warning: could not chmod 600 ${SSH_IDENTITY} (read-only fs / root-squash?); OpenSSH may refuse the key" >&2
            fi
        fi
    fi
}

# Read the node list into ${IPS}: inline [nodes] list wins, else [nodes] file.
load_nodes() {
    IPS=()
    local n=0 ip line

    if (( ${#NODES_INLINE[@]} > 0 )); then
        for ip in "${NODES_INLINE[@]}"; do
            if [[ ! "${ip}" =~ ${IP_RE} ]]; then
                echo "Error: invalid IP '${ip}' in [nodes] list" >&2
                exit 1
            fi
            IPS+=("${ip}")
            n=$((n + 1))
        done
    else
        local file="${NODES_FILE:-${SCRIPT_DIR}/nodes.conf}"
        # Resolve relative paths against UTILS_ROOT.
        [[ "${file}" == /* ]] || file="${UTILS_ROOT}/${file}"
        if [[ ! -r "${file}" ]]; then
            echo "Error: node list file not readable: ${file}" >&2
            exit 1
        fi
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="${line%%#*}"
            ip="$(trim "${line}")"
            [[ -n "${ip}" ]] || continue
            if [[ ! "${ip}" =~ ${IP_RE} ]]; then
                echo "Error: invalid IP '${ip}' in ${file}" >&2
                exit 1
            fi
            IPS+=("${ip}")
            n=$((n + 1))
        done < "${file}"
    fi

    if (( n == 0 )); then
        echo "Error: no nodes configured ([nodes] list or file)" >&2
        exit 1
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --train)
                MODE="train"
                shift
                ;;
            --release)
                MODE="release"
                shift
                ;;
            --kill)
                if [[ -z "${2:-}" ]]; then
                    echo "Usage: $0 [--train|--release|--kill 8,42] [--config FILE] [--exclude 0,1,...]"
                    exit 1
                fi
                MODE="kill"
                IFS=', ' read -r -a KILL_NODE_SUFFIXES <<< "$2"
                shift 2
                ;;
            --config)
                if [[ -z "${2:-}" ]]; then
                    echo "Usage: $0 [--train|--release|--kill 8,42] [--config FILE] [--exclude 0,1,...]"
                    exit 1
                fi
                CONFIG_FILE="$2"
                shift 2
                ;;
            --exclude)
                if [[ -z "${2:-}" ]]; then
                    echo "Usage: $0 [--train|--release|--kill 8,42] [--config FILE] [--exclude 0,1,...]"
                    exit 1
                fi
                IFS=', ' read -r -a EXCLUDE_INDICES <<< "$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: $0 [--train|--release|--kill 8,42] [--config FILE] [--exclude 0,1,...]"
                exit 1
                ;;
        esac
    done
}

is_local() {
    local ip="$1"
    # An empty ip must never classify as local: `hostname -I` output ends
    # with a trailing space, so " ${LOCAL_IPS} " contains a double space
    # that a zero-length pattern would match — routing a bogus command into
    # the LOCAL container (a spurious pkill of OTHER sessions' trainers).
    [[ -n "${ip}" ]] || return 1
    # Quoting "${ip}" makes the dots literal instead of regex wildcards.
    [[ " ${LOCAL_IPS} " =~ [[:space:]]"${ip}"[[:space:]] ]]
}

is_excluded() {
    local ip="$1"
    local idx
    for idx in "${!IPS[@]}"; do
        if [[ "${IPS[$idx]}" == "${ip}" ]]; then
            local e
            for e in "${EXCLUDE_INDICES[@]}"; do
                [[ "${e}" == "${idx}" ]] && return 0
            done
            return 1
        fi
    done
    return 1
}

build_active_ips() {
    ACTIVE_IPS=()
    local ip
    for ip in "${IPS[@]}"; do
        is_excluded "${ip}" || ACTIVE_IPS+=("${ip}")
    done
}

# Quote a command line so the remote shell passes it to `bash -c` as ONE
# argument — immune to single quotes, backslashes and glob characters in the
# command (the old `bash -c '${cmd}'` nesting let the inner quotes cancel out
# and left pattern tokens unquoted/glob-expandable on the remote shell).
remote_bash_cmd() {
    printf 'bash -c %q' "$1"
}

# Run a command inside the target environment (local docker / remote SSH).
run_node() {
    local ip="$1"
    local cmd="$2"

    if is_local "${ip}"; then
        docker exec "${CONTAINER}" bash -c "${cmd}"
    else
        ssh "${SSH_OPTS[@]}" "${ip}" "$(remote_bash_cmd "${cmd}")"
    fi
}

# True if the node answers a trivial command (transport works at all).
node_reachable() {
    run_node "$1" "true" >/dev/null 2>&1
}

# Prints nvidia-smi CSV rows on stdout; exit status is non-zero when the
# command failed (transport failure OR broken driver).
fetch_gpu_data() {
    local ip="$1"
    local cmd rc=0
    cmd='nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
    run_node "${ip}" "${cmd}" 2>/dev/null || rc=$?
    return "${rc}"
}

# Per-IP CUDA-probe verdicts, cached in FILES so an IDLE node is re-probed at
# most once per COOLDOWN instead of re-importing torch over ssh on every
# cycle. Files (not shell variables) are used because manager.sh fetches node
# state in background subshells, where variable mutations are lost; the mtime
# check makes stale files self-expiring after COOLDOWN.
#
# Verdicts: 0 = healthy, 1 = GPU probe failed PROBE_FAIL_THRESHOLD times in a
# row (confirmed BROKEN), 2 = transient/environment failure — NOT treated as
# BROKEN, so a single flaky probe never mislabels a healthy node.
PROBE_CACHE_DIR="${TMPDIR:-/tmp}/utils_monitor_probe-$(id -u)"
PROBE_FAIL_THRESHOLD=3

probe_cuda() {
    local ip="$1"
    # PROBE_COMMAND is an optional env-activation PREFIX (e.g. "source ...
    # && conda activate ENV &&"); the probe script path is derived from
    # workspace_root, never hardcoded in config. torch must be importable by
    # whatever interpreter the prefix selects (system python3 often lacks it).
    local prefix="${PROBE_COMMAND:+${PROBE_COMMAND} }"
    run_node "${ip}" "${prefix}python3 ${CUDA_PROBE}" >/dev/null 2>&1
}

probe_cuda_cached() {
    local ip="$1"
    local now cache
    mkdir -p "${PROBE_CACHE_DIR}" 2>/dev/null
    now=$(date +%s)
    cache="${PROBE_CACHE_DIR}/${ip}"
    if [[ -f "${cache}" ]]; then
        local verdict window
        verdict="$(<"${cache}")"
        case "${verdict}" in
            0|1|2)
                # Confirmed verdicts (0 healthy / 1 BROKEN) stay cached for the
                # full PROBE_INTERVAL (startup + every 10 min by default);
                # inconclusive (2) expires after COOLDOWN so an UNVERIFIED node
                # re-probes soon.
                if (( verdict == 2 )); then
                    window="${COOLDOWN}"
                else
                    window="${PROBE_INTERVAL}"
                fi
                if (( now - $(stat -c '%Y' "${cache}" 2>/dev/null || echo 0) < window )); then
                    return "${verdict}"
                fi
                ;;
            *)
                # Empty/corrupt cache file — Ctrl-C can kill a startup probe
                # subshell between truncate and write, leaving a zero-length
                # file with a FRESH mtime. Treat it as a cache MISS and
                # re-probe: never mislabel a healthy node BROKEN for a full
                # PROBE_INTERVAL on the strength of a truncated file.
                ;;
        esac
    fi

    local verdict=1 fail_file fails
    fail_file="${PROBE_CACHE_DIR}/${ip}.fails"
    if probe_cuda "${ip}"; then
        verdict=0
        rm -f "${fail_file}" 2>/dev/null || true
    else
        local rc=$?
        if (( rc == 1 )); then
            # The probe script itself ran and reported a GPU problem; only
            # PROBE_FAIL_THRESHOLD consecutive failures (one per COOLDOWN)
            # confirm BROKEN. Anything below that is a fluke.
            if [[ -f "${fail_file}" ]]; then
                fails="$(<"${fail_file}")"
            else
                fails=0
            fi
            fails=$((fails + 1))
            printf '%s' "${fails}" > "${fail_file}" 2>/dev/null || true
            if (( fails >= PROBE_FAIL_THRESHOLD )); then
                verdict=1
            else
                verdict=2
            fi
        else
            # Transport (ssh/docker) or interpreter/script-path failure: not a
            # GPU verdict, do not count it — keep the node IDLE.
            verdict=2
        fi
    fi

    # Atomic write (tmp + mv): an interrupted write must never leave a
    # zero-length cache file — the read side treats such a file as a miss
    # and re-probes, so a clean verdict always lands as one rename.
    printf '%s' "${verdict}" > "${cache}.tmp" 2>/dev/null && mv -f "${cache}.tmp" "${cache}" 2>/dev/null || true
    return "${verdict}"
}

trim() {
    local v="$1"
    v="${v#"${v%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"
    printf '%s' "${v}"
}

check_trainer() {
    local ip="$1"
    run_node "${ip}" "pgrep -f '${PGREP_PATTERN}'" >/dev/null 2>&1
}

# Launch the occupancy trainer on a node.
# TRAINER_COMMAND is an optional env-activation PREFIX (e.g. "source ... &&
# conda activate ENV &&"); the launcher script (run_train.sh) is always run and
# locates train.py itself relative to its own dir — so no train.py path is
# hardcoded anywhere. run_train.sh execs the workload with --tag TRAINER_MARKER.
launch_trainer() {
    local ip="$1"
    local prefix="${TRAINER_COMMAND:+${TRAINER_COMMAND} }"

    if is_local "${ip}"; then
        # Log to LOG_FILE (inside the container); braces capture the whole
        # && prefix chain plus the launcher, not just the last command.
        docker exec -d "${CONTAINER}" bash -c \
            "{ ${prefix}TRAIN_TAG='${TRAINER_MARKER}' bash '${LAUNCHER_HOST}'; } > ${LOG_FILE} 2>&1"
    else
        # ssh -f backgrounds the client; setsid puts the wrapper in its own
        # session so the trainer survives the monitor-side connection dropping
        # (keepalive now detects that in seconds). The whole
        # prefix/TRAIN_TAG/launcher chain is wrapped in an inner `bash -c`
        # because setsid execs its first argument directly and cannot run
        # env-assignments or shell builtins (source/conda activate) itself.
        # Single quotes are escaped for the inner shell; the whole line is
        # %q-quoted for the outer ssh shell. The redirects and </dev/null bind
        # to the setsid'd command, so the log stays on the node.
        local remote_script="${WORKSPACE_ROOT}/UTILS/nodes_monitor/run_train.sh"
        local inner="${prefix}TRAIN_TAG='${TRAINER_MARKER}' bash '${remote_script}'"
        local escaped="${inner//\'/\'\\\'\'}"
        ssh -f "${SSH_OPTS[@]}" "${ip}" \
            "$(remote_bash_cmd "setsid bash -c '${escaped}' > ${LOG_FILE} 2>&1 < /dev/null &")"
    fi
}

kill_trainer() {
    local ip="$1"
    # Exits 0 when at least one tagged trainer was killed, 1 when none
    # matched, non-zero (e.g. 255) when the node is unreachable.
    run_node "${ip}" "pkill -f '${PGREP_PATTERN}'" >/dev/null 2>&1
}

get_node_state() {
    local ip="$1"
    declare -n _total_ref="$2"
    declare -n _compute_ref="$3"
    declare -n _mem_used_ref="$4"
    declare -n _trainer_ref="$5"
    declare -n _state_ref="$6"

    _compute_ref=0
    _mem_used_ref=0
    _trainer_ref=0
    _total_ref=-1

    local data rc=0
    data="$(fetch_gpu_data "${ip}")" || rc=$?

    if (( rc != 0 )); then
        # Command failed: unreachable node vs reachable-but-broken (nvidia-smi
        # itself failed) are different problems — report them differently.
        if node_reachable "${ip}"; then
            _state_ref="BROKEN"
        else
            _state_ref="OFFLINE"
        fi
        return
    fi

    if [[ -z "${data}" ]]; then
        _state_ref="OFFLINE"
        return
    fi

    _total_ref=0
    local line
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue

        local idx util used _rest
        IFS=',' read -r idx util used _rest <<< "${line}"

        idx="$(trim "${idx}")"
        util="$(trim "${util}")"
        used="$(trim "${used}")"

        [[ "${util}" =~ ^[0-9]+$ ]] || util=0
        [[ "${used}" =~ ^[0-9]+$ ]] || used=0

        _total_ref=$((_total_ref + 1))

        # Independent counters: a GPU can both compute AND hold memory;
        # USED below means "compute or memory occupied".
        if (( util > COMPUTE_THRESHOLD )); then
            _compute_ref=$((_compute_ref + 1))
        fi
        if (( used > MEM_USED_THRESHOLD )); then
            _mem_used_ref=$((_mem_used_ref + 1))
        fi
    done <<< "${data}"

    if (( _total_ref == 0 )); then
        _state_ref="NO_GPU"
        return
    fi

    if check_trainer "${ip}"; then
        _trainer_ref=1
    fi

    if (( _trainer_ref == 1 )); then
        _state_ref="TRAIN"
    elif (( _compute_ref > 0 || _mem_used_ref > 0 )); then
        _state_ref="USED"
    else
        _state_ref="IDLE"
        # Verdict 1 = PROBE_FAIL_THRESHOLD consecutive GPU-probe failures
        # (confirmed); verdict 2 (transient/environment) stays IDLE. The
        # `|| prc=$?` keeps set -e from aborting on the non-zero verdicts.
        local prc=0
        probe_cuda_cached "${ip}" || prc=$?
        if (( prc == 1 )); then
            _state_ref="BROKEN"
        fi
    fi
}

format_node_line() {
    local ip="$1"
    local total="$2"
    local compute="$3"
    local state="$4"

    local suffix=""
    is_excluded "${ip}" && suffix=" [EXCLUDED]"
    [[ -n "${suffix}" ]] && suffix="${C_GRAY}${suffix}${C_RESET}"

    local state_color="${C_GRAY}"
    case "${state}" in
        IDLE)   state_color="${C_GREEN}" ;;
        TRAIN)  state_color="${C_BLUE}"  ;;
        USED)   state_color="${C_YELLOW}";;
        BROKEN) state_color="${C_RED}"   ;;
    esac

    case "${state}" in
        OFFLINE|NO_GPU|BROKEN)
            printf '%s%-11s%s %s[%s]%s%s\n' \
                "${C_BOLD}" "${ip}" "${C_RESET}" \
                "${state_color}" "${state}" "${C_RESET}" \
                "${suffix}"
            ;;
        *)
            # USED with 0/N computing = memory held but nothing crunching —
            # highlight the counter in red so it stands out.
            local compute_color="${C_RESET}"
            if [[ "${state}" == "USED" && "${compute}" == "0" ]]; then
                compute_color="${C_RED}"
            fi
            printf '%s%-11s%s %s[%s]%s %s%s%s%s\n' \
                "${C_BOLD}" "${ip}" "${C_RESET}" \
                "${state_color}" "${state}" "${C_RESET}" \
                "${compute_color}" "${compute}/${total}" "${C_RESET}" "${suffix}"
            ;;
    esac
}

release_all() {
    echo "Releasing auto-started training jobs on active nodes..."
    local ip rc
    for ip in "${ACTIVE_IPS[@]}"; do
        rc=0
        if kill_trainer "${ip}"; then
            echo "${ip}: released"
        else
            rc=$?
            if (( rc > 1 )); then
                echo "${ip}: node unreachable or command failed (rc=${rc})"
            else
                echo "${ip}: no tagged trainer"
            fi
        fi
    done
}

# Resolve node suffixes (e.g. 8, 42, 118) to full IPs from ${IPS}.
resolve_nodes() {
    KILL_IPS=()
    local suffix ip found
    for suffix in "${KILL_NODE_SUFFIXES[@]}"; do
        [[ "${suffix}" =~ ^[0-9]+$ ]] || {
            echo "Invalid node suffix: ${suffix}" >&2
            exit 1
        }
        found=0
        for ip in "${IPS[@]}"; do
            if [[ "${ip##*.}" == "${suffix}" ]]; then
                KILL_IPS+=("${ip}")
                found=1
                break
            fi
        done
        if (( found == 0 )); then
            echo "Unknown node suffix: ${suffix} (known: ${IPS[*]})" >&2
            exit 1
        fi
    done
}

kill_gpu_procs() {
    local ip="$1"
    local out line rc=0
    out="$(run_node "${ip}" "bash ${KILL_GPU_SCRIPT}" 2>&1)" || rc=$?
    if (( rc != 0 )); then
        echo "${ip}: command failed (rc=${rc}) — node offline or script error"
        return
    fi
    echo "${ip}:"
    while IFS= read -r line; do
        printf '    %s\n' "${line}"
    done <<< "${out}"
}

kill_selected() {
    resolve_nodes
    echo "Killing GPU-occupying processes on: ${KILL_IPS[*]}"
    local ip
    for ip in "${KILL_IPS[@]}"; do
        kill_gpu_procs "${ip}"
    done
}

cleanup() {
    trap - INT TERM EXIT
    # 1. Stop in-flight launchers FIRST so no new spawn can be initiated
    #    after we start killing (a detached ssh -f / docker exec -d child
    #    keeps running even when its launcher subshell is killed).
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
    # 2. ...then kill our trainers twice: a detached ssh -f / docker exec -d
    #    can still land the trainer after the first round (see manager.py).
    local round ip
    for round in 1 2; do
        for ip in ${LAUNCHED_IPS[@]+"${LAUNCHED_IPS[@]}"}; do
            kill_trainer "${ip}" >/dev/null 2>&1 || true
        done
        (( round == 1 )) && sleep 3
    done
    # Clear the screen instead of leaving the last frame behind, then re-show
    # the cursor.
    printf '\033[2J\033[H\033[?25h'
    exit 0
}

main() {
    parse_args "$@"
    load_config
    finalize_config
    load_nodes
    build_active_ips

    if [[ "${MODE}" == "release" ]]; then
        release_all
        exit 0
    fi

    if [[ "${MODE}" == "kill" ]]; then
        kill_selected
        exit 0
    fi

    trap cleanup INT TERM EXIT

    # Startup probe: one health snapshot per node, in parallel, so every node
    # has a verdict before the first tick and the PROBE_INTERVAL cadence
    # (10 min default) starts here. Nodes probed recently (cache fresh) are
    # skipped — never more often than PROBE_INTERVAL per verdict.
    local -a probe_pids=()
    local ip
    for ip in "${ACTIVE_IPS[@]}"; do
        ( probe_cuda_cached "${ip}" >/dev/null 2>&1 ) &
        probe_pids+=($!)
    done
    for p in "${probe_pids[@]}"; do wait "${p}" 2>/dev/null || true; done

    printf '\033[2J\033[H\033[s\033[?25l'

    declare -A last_launch
    local last_action="none"

    while true; do
        local now
        now=$(date +%s)

        local -a lines=()
        lines+=("${C_CYAN}$(date '+%Y-%m-%d %H:%M:%S')${C_RESET}  |  ${C_YELLOW}last action: ${last_action}${C_RESET}")
        lines+=("${C_GRAY}----------------------------------------${C_RESET}")

        local launched_now=""

        for ip in "${IPS[@]}"; do
            local total=0 compute=0 mem_used=0 trainer=0 state="OFFLINE"

            get_node_state "${ip}" total compute mem_used trainer state
            lines+=("$(format_node_line "${ip}" "${total}" "${compute}" "${state}")")

            if [[ "${MODE}" == "train" ]] && ! is_excluded "${ip}" && [[ "${state}" == "IDLE" ]]; then
                local prev=${last_launch["${ip}"]:-0}
                if (( now - prev > COOLDOWN )); then
                    LAUNCHED_IPS+=("${ip}")
                    launch_trainer "${ip}" &
                    last_launch["${ip}"]=${now}
                    launched_now+="${ip} "
                fi
            fi
        done

        wait
        [[ -n "${launched_now}" ]] && last_action="started training on ${launched_now% }"

        printf '\033[u\033[J'
        printf '%s\n' "${lines[@]}"

        sleep "${INTERVAL}"
    done
}

# Run standalone; when sourced (manager.sh) only the helpers are used.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    main "$@"
fi
