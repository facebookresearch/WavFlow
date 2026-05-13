#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# shellcheck disable=SC1091 source=./_run.sh
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh"

if [[ -z "${NNODES:-}" || -z "${NODE_RANK:-}" || -z "${MASTER_ADDR:-}" ]]; then
    echo "ERROR: NNODES, NODE_RANK and MASTER_ADDR must all be set." >&2
    exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG_PATH="${CONFIG_PATH:-wavflow/configs/train.yaml}"

torchrun \
    --nnodes="${NNODES}" \
    --node-rank="${NODE_RANK}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    -m wavflow.train \
    --config "${CONFIG_PATH}" \
    "$@"
