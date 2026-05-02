#!/usr/bin/env bash
# Smoke-test launcher: runs vanilla SeeKer for 1 epoch on ShanghaiTech to
# verify the data pipeline + device patches work end-to-end. Not for benchmarking.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

cd "${PROJECT_ROOT}/seeker"

export PYTHONUNBUFFERED=1

"${PYTHON}" -u seeker.py \
    --dataset ShanghaiTech \
    --data_dir "${PROJECT_ROOT}/data" \
    --exp_dir "${PROJECT_ROOT}/exp_dir" \
    --device mps \
    --num_workers 0 \
    --epochs 1 \
    --batch_size 256 \
    --seg_len 24 \
    --seed 42 \
    "$@"
