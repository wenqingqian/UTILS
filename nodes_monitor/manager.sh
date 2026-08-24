#!/usr/bin/env bash
# Thin entry point for the node manager.
#
# The old bash workbench that used to live in this file is gone — all real
# logic now lives in the python package under utils/manager/. This script only
# finds a usable interpreter and dispatches:
#
#   ./manager.sh                                  interactive GUI (default)
#   ./manager.sh --gui [ignored...]               force the GUI — ALL other
#                                                 arguments are ignored (spec:
#                                                 --gui short-circuits everything)
#   ./manager.sh --train [all|1,2] [--config F]   one-shot: occupy currently-IDLE
#                                                 target nodes now, print the
#                                                 result, exit (no TUI)
#   ./manager.sh --kill all|1,2                   one-shot: kill ALL GPU processes
#                                                 on the targets (gpu_kill.sh)
#   ./manager.sh --release [all|1,2]              one-shot: kill only our tagged
#                                                 trainers, leave everything else
#   ./manager.sh --help                           show this text
#
# Targets are `all` or comma-separated 1-based node indices (see manager_cli.py).
#
# Environment:
#   PYTHON                 interpreter to use; bypasses auto-detection
#   MANAGER_SH_DRY_RUN=1   print the resolved python + command line instead of
#                          exec'ing — makes this wrapper testable without
#                          starting a TUI or touching the cluster

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GUI_PY="${SCRIPT_DIR}/utils/manager/manager.py"      # interactive TUI
CLI_PY="${SCRIPT_DIR}/utils/manager/manager_cli.py"  # one-shot commands
VENV_PY="/public-nvme/gzlu/.venv/bin/python3"  # project fallback interpreter

usage() {
    cat <<'EOF'
usage:
  ./manager.sh                                  interactive GUI (default)
  ./manager.sh --gui [ignored...]               force the GUI, ignore ALL other args
  ./manager.sh --train [all|1,2] [--config F]   one-shot: occupy currently-IDLE target nodes now
  ./manager.sh --kill all|1,2                   one-shot: kill ALL GPU processes on the targets (gpu_kill.sh)
  ./manager.sh --release [all|1,2]              one-shot: kill only our tagged trainers
  ./manager.sh --help                           this text

one-shot commands print their result and exit — no TUI is started.
targets: `all` or comma-separated 1-based node indices.
env: PYTHON (interpreter override), MANAGER_SH_DRY_RUN=1 (print, don't exec)
EOF
}

# ---------------------------------------------------------------------------
# Python resolution
# ---------------------------------------------------------------------------

# tomllib is stdlib only since python 3.11 and the config is TOML, so this
# import is the cheapest possible capability probe for a usable interpreter.
has_tomllib() {
    "$1" -c 'import tomllib' >/dev/null 2>&1
}

resolve_python() {
    # 1) explicit override wins — the caller vouches for it being >= 3.11
    if [[ -n "${PYTHON:-}" ]]; then
        printf '%s\n' "${PYTHON}"
        return 0
    fi
    # 2) python3 on PATH, if new enough (on this box it is 3.10 -> rejected)
    if has_tomllib python3; then
        printf '%s\n' python3
        return 0
    fi
    # 3) project venv fallback — the system python3 may be < 3.11 (no tomllib)
    if [[ -x "${VENV_PY}" ]] && has_tomllib "${VENV_PY}"; then
        printf '%s\n' "${VENV_PY}"
        return 0
    fi
    echo "manager.sh needs python3 >= 3.11 (tomllib) — activate /public-nvme/gzlu/.venv or set PYTHON" >&2
    return 1
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Resolve the interpreter, make sure both python entry points are actually in
# place, then hand the process over with exec so signals (Ctrl-C, TERM) reach
# python directly instead of a lingering bash parent.
launch() {
    local target="$1"; shift
    local py
    py="$(resolve_python)"   # set -e aborts here when resolution failed

    local f
    for f in "${GUI_PY}" "${CLI_PY}"; do
        if [[ ! -f "${f}" ]]; then
            echo "manager.sh: ${f} not found — the manager python package is missing or incomplete" >&2
            exit 1
        fi
    done

    if [[ "${MANAGER_SH_DRY_RUN:-0}" == "1" ]]; then
        printf '[dry-run] would exec:'
        printf ' %q' "${py}" "${target}" "$@"
        printf '\n'
        exit 0
    fi
    exec "${py}" "${target}" "$@"
}

# A --gui anywhere on the command line wins and short-circuits to the GUI;
# per spec ALL other arguments are ignored in that case (even --config).
gui=0
for arg in "$@"; do
    if [[ "${arg}" == "--gui" ]]; then
        gui=1
        break
    fi
done

if (( gui )) || (( $# == 0 )); then
    launch "${GUI_PY}"
elif (( $# == 1 )) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
else
    # one-shot: manager_cli.py prints its result and exits — no TUI is started
    launch "${CLI_PY}" "$@"
fi
