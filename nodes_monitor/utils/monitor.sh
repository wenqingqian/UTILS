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
SSH_IDENTITY=""        # [ssh] identity: key path RELATIVE to workspace_root (raw config value)
SSH_IDENTITY_PATH=""   # resolved key path as THIS machine sees it (set by finalize_config)
SSH_PORT=2222
WORKSPACE_ROOT=""      # [paths] workspace_root: HOST view of the workspace dir
HOST_IDENTITY=""       # <workspace_root>/<identity> — where the ssh client reads the key
NODE_IDENTITY=""       # <NODE_ROOT>/<identity> — the same key as a node sees it
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

IPS=()                 # node specs as configured: ip or ip:port
NODE_HOSTS=()          # bare IP for transport/locality checks
NODE_PORTS=()          # per-node port; 0 means use SSH_PORT
IP_RE='^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}$'
NODE_RE='^((25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3})(:([1-9][0-9]{0,4}))?$'

# Where the workspace is mounted for commands running ON A NODE: the container
# of the local node (docker exec) and the remote nodes (plain ssh) both see the
# directory as /workspace, while THIS host — where the tools run, and hence
# where the ssh client reads the key — sees it as [paths] workspace_root. The
# node-side mount point is the docker deployment's convention, not a config
# value; only the host view is configurable.
NODE_ROOT="/workspace"

# Node-side CUDA probe path (derived by finalize_config from NODE_ROOT + this
# repo's path relative to the workspace — the repo may sit in a workspace
# subdirectory, so no fixed "UTILS" segment). It
# runs inside the container environment on the local node (docker exec) and via
# plain ssh on the remote nodes. The probe's interpreter is configurable via
# [trainer] probe_command (the system python3 may not see torch on nodes where
# it lives in a conda env).
CUDA_PROBE=""

# Tuning knobs; all overridable via the [monitor] section of the config.
INTERVAL=5
# Trust window for cached CUDA-probe verdicts. The probe occupies GPU memory
# while it runs (a CUDA context per GPU, visible in nvidia-smi), so it must
# not fire casually: within COOLDOWN an earlier verdict is reused verbatim.
# Default 12h.
COOLDOWN=43200
# Inconclusive (2) verdicts expire after PROBE_RETRY instead — re-probing
# them touches no GPU (their failures happen at the transport/environment
# layer), and this cadence also paces BROKEN confirmation
# (PROBE_FAIL_THRESHOLD consecutive rc=1 failures, ~3 min apart by default).
PROBE_RETRY=60

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

    # ---- workspace path model ----
    # The same workspace directory has two names:
    #   host view — [paths] workspace_root: where this script (and hence the
    #               ssh client reading the key) runs;
    #   node view — ${NODE_ROOT}: every command sent to a node runs in this
    #               view (docker exec on the local node, plain ssh on the
    #               remote ones).
    # [ssh] identity is stored RELATIVE to the workspace, so the key file is
    # <workspace_root>/<identity> here and ${NODE_ROOT}/<identity> on a node.
    while [[ "${WORKSPACE_ROOT}" == */ && "${WORKSPACE_ROOT}" != "/" ]]; do
        WORKSPACE_ROOT="${WORKSPACE_ROOT%/}"
    done
    if [[ "${SSH_IDENTITY}" == /* ]]; then
        echo "Error: [ssh] identity must be relative to workspace_root (got '${SSH_IDENTITY}'; use e.g. identity = \"./cluster_ssh_key\")" >&2
        exit 1
    fi
    local rel_identity="${SSH_IDENTITY}"
    while [[ "${rel_identity}" == ./* ]]; do rel_identity="${rel_identity#./}"; done
    if [[ -z "${rel_identity}" ]]; then
        echo "Error: [ssh] identity must name a key file (got '${SSH_IDENTITY}')" >&2
        exit 1
    fi
    HOST_IDENTITY="${WORKSPACE_ROOT}/${rel_identity}"
    NODE_IDENTITY="${NODE_ROOT}/${rel_identity}"
    # Resolve the key against THIS machine's view: normally the host view
    # exists; when the tools themselves run inside the container, only the
    # node view does. SSH_IDENTITY (the raw config value) stays untouched so a
    # repeated finalize_config call re-resolves from the config, not from its
    # own output.
    if [[ -f "${HOST_IDENTITY}" ]]; then
        SSH_IDENTITY_PATH="${HOST_IDENTITY}"
    elif [[ -f "${NODE_IDENTITY}" ]]; then
        SSH_IDENTITY_PATH="${NODE_IDENTITY}"
    else
        echo "Error: [ssh] identity file not found: tried ${HOST_IDENTITY} and ${NODE_IDENTITY}" >&2
        exit 1
    fi

    # CUDA probe path as seen from a node (docker exec on the local node; the
    # remote branch runs the same node-side path over ssh). The repo's location
    # within the workspace is DERIVED from this script's real path — never a
    # hardcoded "UTILS" segment: the repo may sit in a subdirectory of the
    # workspace (e.g. <ws>/sga_framework/UTILS), where a fixed segment resolves
    # to a path that does not exist on any node. Host view first, node view as
    # fallback (the tools themselves may run inside the container); pwd -P and
    # realpath collapse symlinks so the relative path compares physical dirs.
    local utils_root utils_rel="" root
    utils_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
    for root in "${WORKSPACE_ROOT}" "${NODE_ROOT}"; do
        utils_rel="$(realpath -m --relative-to="${root}" "${utils_root}")"
        if [[ "${utils_rel}" != ".." && "${utils_rel}" != ../* ]]; then
            break
        fi
    done
    if [[ "${utils_rel}" == ".." || "${utils_rel}" == ../* ]]; then
        echo "Error: the UTILS repo (${utils_root}) is outside both [paths] workspace_root (${WORKSPACE_ROOT}) and the node workspace (${NODE_ROOT}) — move it under the workspace" >&2
        exit 1
    fi
    CUDA_PROBE="${NODE_ROOT}/${utils_rel}/nodes_monitor/utils/cuda_probe.py"

    # Auto-fix SSH private key permissions: OpenSSH ignores keys that are
    # group/world-readable. Tighten to 600 if too open so auth does not fail.
    # (GNU-specific stat -c; the 8# base forces octal interpretation.)
    local perm
    perm="$(stat -c '%a' "${SSH_IDENTITY_PATH}" 2>/dev/null || echo "")"
    if [[ -n "${perm}" && $(( 8#${perm} & 077 )) -ne 0 ]]; then
        if ! chmod 600 "${SSH_IDENTITY_PATH}" 2>/dev/null; then
            echo "Warning: could not chmod 600 ${SSH_IDENTITY_PATH} (read-only fs / root-squash?); OpenSSH may refuse the key" >&2
        fi
    fi
}

# Read the node list into ${IPS}: inline [nodes] list wins, else [nodes] file.
load_nodes() {
    IPS=()
    NODE_HOSTS=()
    NODE_PORTS=()
    local n=0 spec host port line

    if (( ${#NODES_INLINE[@]} > 0 )); then
        for spec in "${NODES_INLINE[@]}"; do
            if [[ ! "${spec}" =~ ${NODE_RE} ]]; then
                echo "Error: invalid node '${spec}' in [nodes] list (expected IP or IP:port)" >&2
                exit 1
            fi
            host="${BASH_REMATCH[1]}"
            # Group 6 = port digits (empty for a bare IP -> 0 = use SSH_PORT).
            port="${BASH_REMATCH[6]:-0}"
            (( port == 0 || port <= 65535 )) || { echo "Error: invalid port in node '${spec}'" >&2; exit 1; }
            IPS+=("${spec}")
            NODE_HOSTS+=("${host}")
            NODE_PORTS+=("${port}")
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
            spec="$(trim "${line}")"
            [[ -n "${spec}" ]] || continue
            if [[ ! "${spec}" =~ ${NODE_RE} ]]; then
                echo "Error: invalid node '${spec}' in ${file} (expected IP or IP:port)" >&2
                exit 1
            fi
            host="${BASH_REMATCH[1]}"
            # Group 6 = port digits (empty for a bare IP -> 0 = use SSH_PORT).
            port="${BASH_REMATCH[6]:-0}"
            (( port == 0 || port <= 65535 )) || { echo "Error: invalid port in node '${spec}'" >&2; exit 1; }
            IPS+=("${spec}")
            NODE_HOSTS+=("${host}")
            NODE_PORTS+=("${port}")
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

node_host() {
    local spec="$1"
    if [[ "${spec}" =~ ${NODE_RE} ]]; then printf '%s' "${BASH_REMATCH[1]}"; else printf '%s' "${spec%%:*}"; fi
}

node_port() {
    local spec="$1"
    if [[ "${spec}" =~ ${NODE_RE} ]]; then
        # Group 6 is the port DIGITS (group 5 wraps them together with the
        # ':'), and it is EMPTY for the common bare-IP entry. It must fall
        # back to [ssh].port — never to 0: `ssh -p 0` dies with "Bad port
        # '0'", so every node without a per-node port failed transport
        # (view.sh saw the whole cluster as OFFLINE while manager.py, whose
        # parse_node() defaults correctly, saw them as IDLE).
        local port="${BASH_REMATCH[6]:-}"
        [[ -n "${port}" ]] || port="${SSH_PORT}"
        (( port <= 65535 )) || port="${SSH_PORT}"
        printf '%s' "${port}"
    else
        printf '%s' "${SSH_PORT}"
    fi
}

is_local() {
    local spec="$1" ip
    ip="$(node_host "${spec}")"
    [[ -n "${ip}" ]] || return 1
    [[ " ${LOCAL_IPS} " =~ [[:space:]]"${ip}"[[:space:]] ]]
}

# Quote a command line so the remote shell passes it to `bash -c` as ONE
# argument — immune to single quotes, backslashes and glob characters in the
# command (the old `bash -c '${cmd}'` nesting let the inner quotes cancel out
# and left pattern tokens unquoted/glob-expandable on the remote shell).
remote_bash_cmd() {
    printf 'bash -c %q' "$1"
}

# Command timeout for every node contact: a wedged node (nvidia-smi hung on
# a stuck driver) must delay a query by seconds, not forever — and the
# timeout also bounds how long an interrupted view.sh's children can linger.
# Matches the python manager's run_node timeout. -k 5: a client that ignores
# TERM is KILLed 5s later; exit 124 is just another transport failure to the
# callers.
RUN_NODE_TIMEOUT=30

# Build the ssh options for one node. -i points at the key resolved by
# finalize_config (SSH_IDENTITY_PATH). NOTE: StrictHostKeyChecking=no +
# UserKnownHostsFile=/dev/null disable host-key verification — an accepted
# MITM exposure for internal cluster tooling behind a jump host.
# ServerAliveInterval/ServerAliveCountMax detect a node dying mid-command
# (~15s) so a strictly sequential query loop cannot stall on TCP
# retransmission timeouts.
node_ssh_opts() {
    local spec="$1" port
    port="$(node_port "${spec}")"
    NODE_SSH_OPTS=(
        -p "${port}" -i "${SSH_IDENTITY_PATH}"
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
        -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3
        -o BatchMode=yes -o LogLevel=ERROR
    )
}

# Run a command inside the target environment (local docker / remote SSH).
run_node() {
    local ip="$1"
    local cmd="$2"

    if is_local "${ip}"; then
        timeout -k 5 "${RUN_NODE_TIMEOUT}" docker exec "${CONTAINER}" bash -c "${cmd}"
    else
        node_ssh_opts "${ip}"
        timeout -k 5 "${RUN_NODE_TIMEOUT}" ssh "${NODE_SSH_OPTS[@]}" "$(node_host "${ip}")" "$(remote_bash_cmd "${cmd}")"
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

# Per-IP CUDA-probe verdicts, cached in FILES so a view.sh call can reuse a
# confirmed BROKEN result without re-importing torch over ssh. The probe
# occupies GPU memory while it runs, so each view.sh run probes at most once per
# GPU-bearing node. Files (not shell variables) are used because view.sh fetches
# node state in background subshells, where variable mutations are lost.
#
# Only verdict 1 (confirmed BROKEN) is reusable across calls. Healthy (0) and
# inconclusive (2) results are deliberately not cached as trusted outcomes, so
# every subsequent view can re-check a node that may have recovered or become
# wedged after the previous observation.

PROBE_CACHE_DIR="${TMPDIR:-/tmp}/utils_monitor_probe-$(id -u)"
PROBE_FAIL_THRESHOLD=3

probe_cuda() {
    local ip="$1"
    # PROBE_COMMAND is an optional env-activation PREFIX (e.g. "source ...
    # && conda activate ENV &&"); the probe script path is derived from the
    # node-side workspace mount point (NODE_ROOT), never hardcoded in config.
    # torch must be importable by whatever interpreter the prefix selects
    # (system python3 often lacks it).
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
        local verdict
        verdict="$(<"${cache}")"
        # Only a confirmed BROKEN verdict is trusted across view.sh calls.
        # Healthy and inconclusive verdicts are observations for that call, not
        # durable evidence that the node remains healthy.
        if [[ "${verdict}" == "1" ]]; then
            if (( now - $(stat -c '%Y' "${cache}" 2>/dev/null || echo 0) < COOLDOWN )); then
                return 1
            fi
        fi
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
            # PROBE_FAIL_THRESHOLD consecutive failures (one per PROBE_RETRY)
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

    # Probe every GPU-bearing node, including nodes that currently hold memory
    # or run compute. A CUDA context failure is a health property and must not
    # be hidden behind the occupancy-derived USED state. Only a confirmed
    # BROKEN verdict overrides occupancy; an inconclusive probe preserves
    # TRAIN/USED when work is visible, but keeps an otherwise-free node out of
    # the IDLE pool as UNVERIFIED.
    local prc=0
    probe_cuda_cached "${ip}" || prc=$?
    if (( prc == 1 )); then
        _state_ref="BROKEN"
    elif (( _trainer_ref == 1 )); then
        _state_ref="TRAIN"
    elif (( _compute_ref > 0 || _mem_used_ref > 0 )); then
        _state_ref="USED"
    elif (( prc == 2 )); then
        _state_ref="UNVERIFIED"
    else
        _state_ref="IDLE"
    fi
}

# Source-only library: refuse direct execution. Everything above is function
# and variable definitions, so sourcing runs nothing and leaves the caller's
# set flags untouched.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "monitor.sh is the shared nodes_monitor plumbing (sourced by view.sh), not a launcher — use view.sh or manager.sh" >&2
    exit 1
fi
