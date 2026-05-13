#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# shellcheck disable=SC1091 source=./_run.sh
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CONFIG_PATH="${CONFIG_PATH:-feature_extract/configs/extract_t2a.yaml}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m feature_extract.extract_t2a_pth \
    --config "${CONFIG_PATH}" \
    "$@"
