#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# shellcheck disable=SC1091 source=./_run.sh
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh"

GPU="${GPU:-0}"
CONFIG_PATH="${CONFIG_PATH:-wavflow/configs/infer.yaml}"

# ---- Parse a few convenience flags (everything else falls through) ----
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)    GPU="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        *)        EXTRA+=("$1"); shift ;;
    esac
done

CUDA_VISIBLE_DEVICES="$GPU" python -m wavflow.infer \
    --config "${CONFIG_PATH}" \
    "${EXTRA[@]}"
