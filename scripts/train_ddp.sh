#!/usr/bin/env bash
# 多卡训练（torchrun 启动）。单卡直接不用它。
#   NPROC=4 bash scripts/train_ddp.sh
#   NPROC=4 EXTRA="--epochs 60" bash scripts/train_ddp.sh
set -euo pipefail

NPROC="${NPROC:-1}"
CONFIG="${CONFIG:-configs/base.yaml}"
OUT="${OUT:-runs/base}"
EXTRA="${EXTRA:-}"

if [ "$NPROC" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NPROC" -m lane_refine.train \
        --config "$CONFIG" --out "$OUT" $EXTRA
else
    python -m lane_refine.train --config "$CONFIG" --out "$OUT" $EXTRA
fi
