#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Auto-activate conda env if requested and not already active.
WAVFLOW_ENV="${WAVFLOW_ENV:-wavflow}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "$WAVFLOW_ENV" ]]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        if ! eval "$(conda shell.bash hook 2>/dev/null)"; then
            CONDA_BASE="$(conda info --base 2>/dev/null || true)"
            if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
                source "$CONDA_BASE/etc/profile.d/conda.sh"
            fi
        fi
        if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$WAVFLOW_ENV"; then
            conda activate "$WAVFLOW_ENV"
        fi
    fi
fi

echo "[wavflow] REPO_ROOT=$REPO_ROOT  ENV=${CONDA_DEFAULT_ENV:-system}  NPROC_PER_NODE=$NPROC_PER_NODE"
