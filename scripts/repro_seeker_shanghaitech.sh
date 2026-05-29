#!/usr/bin/env bash
# Full vanilla-SeeKer reproduction on ShanghaiTech, targeting the paper's
# 0.855 AUROC. Batch size held at 256 (smoke-test's proven MPS-fitting size;
# paper uses 1024 on CUDA, which exceeds M-series unified memory budgets).
# Expected wall-clock: ~3h on Apple Silicon MPS.
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
    --epochs 10 \
    --batch_size 256 \
    --seg_len 24 \
    --seed 42 \
    "$@"
