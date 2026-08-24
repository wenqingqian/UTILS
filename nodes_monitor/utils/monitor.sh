#!/usr/bin/env bash
# Shared bash plumbing for the nodes_monitor tools — sourced by view.sh (the
# read-only node-state viewer). This is NOT a launcher: executed directly it
# prints an error and exits 1 (guard at the bottom). Use view.sh for read-only
# queries and manager.sh for management actions (occupy/kill/release — those
# live in the python manager now, no bash counterpart remains here).
#
# Config resolution order (highest first):
#   1. --config FILE — a pre-set $CONFIG_FILE (view.sh hands its --config over
#      after sourcing, before load_config runs)
#   2. ../.data/nodes_monitor/config.toml   (gitignored, the real/debug config)
#   3. nodes_monitor/config.toml            (committed template, placeholder values)
#
# Requires python3 >= 3.11 (tomllib) on the machine sourcing this file;
# resolved via $PYTHON, then PATH, then the /public-nvme/gzlu/.venv fallback
# (see _toml_python).
#
# Sourcing contract: sourcing this file RUNS NOTHING — only the helpers below
# are defined — and it never changes the caller's set flags (no set
# -e/-u/-o pipefail anywhere here; callers must set their own flags first).

# Path anchors — this file lives one level deeper than the files it serves:
#   SCRIPT_DIR  = .../nodes_monitor/utils   (this file's own dir)
#   MON_DIR     = .../nodes_monitor         (config.toml / nodes.conf live here)
#   UTILS_ROOT  = .../UTILS                 (the .data config override lives here)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UTILS_ROOT="$(cd "${MON_DIR}/.." && pwd)"

CONFIG_FILE=""

# ---------------------------------------------------------------------------
# Sectioned TOML loading (embedded python tomllib -> KEY=value -> bash source)
# ---------------------------------------------------------------------------

# Python interpreter for TOML parsing — needs >= 3.11 (tomllib is stdlib only
# since 3.11, so importing it is the cheapest capability probe). Resolution
# mirrors manager.sh: $PYTHON wins, then a python3 on PATH with tomllib, then
# the project venv fallback (the system python3 may be too old — on this box
# it is 3.10).
_toml_python() {
    if [[ -n "${PYTHON:-}" ]]; then
        printf '%s\n' "${PYTHON}"
        return 0
    fi
    if python3 -c 'import tomllib' 2>/dev/null; then
        printf '%s\n' python3
        return 0
    fi
    local venv_py="/public-nvme/gzlu/.venv/bin/python3"
    if [[ -x "${venv_py}" ]] && "${venv_py}" -c 'import tomllib' 2>/dev/null; then
        printf '%s\n' "${venv_py}"
        return 0
    fi
    return 1
}

# Parse the TOML into flat KEY=value lines (sections joined with '_', lists as
# newline-separated KEY+=item) on stdout. Requires python3 >= 3.11 (tomllib).
# Values are emitted raw (no quoting/escaping) so bash reads them verbatim;
# the only protocol limitation is that values must not contain newlines.
_toml_to_env() {
    local file="$1"
    "${TOML_PY}" - "$file" <<'PY'
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
    local tmpl_cfg="${MON_DIR}/config.toml"
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
# Every key load_config reads is declared here — the schema is shared with the
# python manager even where this bash side no longer consumes the value
# (LAUNCHER_HOST/TRAINER_COMMAND/THEME are read so unknown-key warnings stay
# accurate and both sides parse the same config).
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

# Node-side CUDA probe (in-container path, derived from workspace_root by
# finalize_config). It runs inside the container environment on the local node
# (docker exec) and via plain ssh on remote nodes — which requires the
# workspace to be mounted at workspace_root on the remote hosts as well. The
# probe's interpreter is configurable via [trainer] probe_command (R3: the
# system python3 may not see torch on nodes where it lives in a conda env).
CUDA_PROBE=""

# Tuning knobs; all overridable via the [monitor] section of the config.
INTERVAL=5
COOLDOWN=60
# CUDA health-probe cadence: confirmed verdicts are re-probed every
# PROBE_INTERVAL. Inconclusive probes expire after COOLDOWN instead.
PROBE_INTERVAL=600

COMPUTE_THRESHOLD=0
MEM_USED_THRESHOLD=100
LOG_FILE="/tmp/utils_train.log"

LOCAL_IPS="$(hostname -I 2>/dev/null || hostname -i 2>/dev/null || true)"

load_config() {
    resolve_config
    local line toml_env
    # Resolve the TOML-parsing interpreter once (see _toml_python); TOML_PY is
    # intentionally a global so a second load_config call skips the probes.
    # Fail loudly instead of silently keeping defaults when no usable
    # interpreter exists or the parse itself fails (TOML syntax error).
    if [[ -z "${TOML_PY:-}" ]]; then
        if ! TOML_PY="$(_toml_python)"; then
            echo "Error: need python3 >= 3.11 (tomllib) to parse ${CONFIG_FILE} — activate /public-nvme/gzlu/.venv or set PYTHON" >&2
            exit 1
        fi
    fi
    if ! toml_env="$(_toml_to_env "${CONFIG_FILE}")"; then
        echo "Error: failed to parse config: ${CONFIG_FILE} (TOML syntax error?)" >&2
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

    # CUDA probe as seen INSIDE the container (docker exec on the local node;
    # the remote branch runs the same in-container path over ssh). Derived
    # from workspace_root so it lives in the same mount point on every node.
    CUDA_PROBE="${WORKSPACE_ROOT}/UTILS/nodes_monitor/utils/cuda_probe.py"

    # NOTE: StrictHostKeyChecking=no + UserKnownHostsFile=/dev/null disable
    # host-key verification — an accepted MITM exposure for internal cluster
    # tooling behind a jump host.
    #
    # ServerAliveInterval/ServerAliveCountMax detect a node dying mid-command
    # (~15s) so a strictly sequential query loop cannot stall on TCP
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
        local file="${NODES_FILE:-${MON_DIR}/nodes.conf}"
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

trim() {
    local v="$1"
    v="${v#"${v%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"
    printf '%s' "${v}"
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
# query. Files (not shell variables) are used because view.sh fetches node
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
                # full PROBE_INTERVAL (10 min by default); inconclusive (2)
                # expires after COOLDOWN so an UNVERIFIED node re-probes soon.
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
                # Empty/corrupt cache file — Ctrl-C can kill a probe subshell
                # between truncate and write, leaving a zero-length file with
                # a FRESH mtime. Treat it as a cache MISS and re-probe: never
                # mislabel a healthy node BROKEN for a full PROBE_INTERVAL on
                # the strength of a truncated file.
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

check_trainer() {
    local ip="$1"
    run_node "${ip}" "pgrep -f '${PGREP_PATTERN}'" >/dev/null 2>&1
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
        # (confirmed BROKEN). Verdict 2 (transient/environment) now yields
        # UNVERIFIED instead of IDLE — same semantics as manager.py's
        # get_node_state, whose --train auto-launch only fires on state ==
        # IDLE: an unverified (inconclusively probed) node must never report
        # IDLE, or it would keep being auto-occupied.
        # The `|| prc=$?` keeps set -e from aborting on the non-zero verdicts.
        local prc=0
        probe_cuda_cached "${ip}" || prc=$?
        if (( prc == 1 )); then
            _state_ref="BROKEN"
        elif (( prc == 2 )); then
            _state_ref="UNVERIFIED"
        fi
    fi
}

# Source-only library: refuse direct execution. Everything above is function
# and variable definitions, so sourcing runs nothing and leaves the caller's
# set flags untouched.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "monitor.sh is the shared nodes_monitor plumbing (sourced by view.sh), not a launcher — use view.sh or manager.sh" >&2
    exit 1
fi
