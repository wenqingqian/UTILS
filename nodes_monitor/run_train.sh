#!/usr/bin/env bash
# Fallback trainer launcher, used only when [trainer] command is NOT set in
# config.toml. The preferred, fully-configurable path is to set [trainer]
# command directly (environment activation, paths, tag all live there).
#
# Customization via environment variables:
#   TRAIN_COMMAND  : full command line to exec (highest priority).
#   TRAIN_PY       : path to the workload script (default: train.py next to
#                    this script, i.e. this directory's occupancy simulator).
#   TRAIN_TAG      : process tag             (default: __UTILS_train_job__).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${TRAIN_COMMAND:-}" ]]; then
    exec bash -c "${TRAIN_COMMAND}"
fi

TRAIN_PY="${TRAIN_PY:-${SCRIPT_DIR}/train.py}"
TRAIN_TAG="${TRAIN_TAG:-__UTILS_train_job__}"

exec "${PYTHON:-python3}" "${TRAIN_PY}" --tag "${TRAIN_TAG}"
