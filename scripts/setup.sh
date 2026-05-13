#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

set -euo pipefail

ENV_NAME="${1:-wavflow}"
PYTHON_VER="${PYTHON_VER:-3.10}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[wavflow] Setting up env '$ENV_NAME' (python=$PYTHON_VER) in $REPO_ROOT"

# 1. Make sure conda is available.
if ! command -v conda >/dev/null 2>&1; then
    echo "[wavflow] ERROR: conda not found. Install miniconda first:" >&2
    echo "    https://docs.conda.io/projects/miniconda/en/latest/" >&2
    exit 1
fi

# Source conda init so `conda activate` works in this script. We try the
# modern `conda shell.bash hook` first (works regardless of install location),
# then fall back to sourcing the profile script if that fails.
# shellcheck disable=SC1091
if ! eval "$(conda shell.bash hook 2>/dev/null)"; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "[wavflow] ERROR: cannot initialize conda in this shell." >&2
        echo "[wavflow] Try running 'conda init bash' once and re-open your shell." >&2
        exit 1
    fi
fi

# 2. Create the conda env if it doesn't exist.
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[wavflow] Creating conda env '$ENV_NAME' ..."
    conda create -n "$ENV_NAME" "python=$PYTHON_VER" -y
fi
conda activate "$ENV_NAME"

# 3. Install Python dependencies first (pinned versions for reproducibility).
echo "[wavflow] Installing requirements ..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Editable install of the package itself. --no-deps because all deps are
#    already satisfied by requirements.txt; this avoids accidental version
#    bumps from the looser specs in pyproject.toml.
pip install -e . --no-deps

# 5. Finally install ffmpeg<7 via conda (torio's native backend needs ffmpeg 4-6,
#    not 7+). Done last so it can't disturb the pip-resolved torch stack.
echo "[wavflow] Installing ffmpeg (<7) into '$ENV_NAME' ..."
conda install -n "$ENV_NAME" -c conda-forge "ffmpeg<7" -y

echo
echo "[wavflow] Done. Activate the environment with:"
echo "    conda activate $ENV_NAME"
echo
echo "Then try:"
echo "    bash scripts/launch/train_single_node.sh --help"
echo "    bash scripts/launch/predict.sh --help"
