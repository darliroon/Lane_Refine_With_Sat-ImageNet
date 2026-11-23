#!/usr/bin/env bash
# 一键搭环境。幂等，可重复执行。
set -euo pipefail

PYTHON="${PYTHON:-python}"
MIRROR="${MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
VENV="${VENV:-.venv}"

echo "==> Python: $($PYTHON -V)"

if [ ! -d "$VENV" ]; then
    echo "==> 创建虚拟环境 $VENV"
    $PYTHON -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "==> 使用 $(which python)"

pip install -q --upgrade pip -i "$MIRROR"

echo "==> 安装依赖（torch 约 2 GB，首次较慢）"
pip install -r requirements.txt -i "$MIRROR"

echo "==> 环境自检"
PYTHON=python bash scripts/check_env.sh

cat <<'EOF'

======================== 完成 ========================
后续步骤：
  1. 合成数据自检（推荐先跑，验证代码路径）
       python scripts/make_synthetic.py --out data_synth --n 24
       python -m lane_refine.train --config configs/base.yaml \
         --manifest data_synth/train.jsonl --val_manifest data_synth/val.jsonl \
         --out runs/synth --epochs 8

  2. 放真实数据后生成清单
       python scripts/prepare_manifest.py --b data/B --a data/A --gt data/GT \
         --out data/train.jsonl --split-train 0.9

  3. 训练
       python -m lane_refine.train --config configs/base.yaml --out runs/base

  4. 推理
       python -m lane_refine.infer --ckpt runs/base/best.pt \
         --rgb xx_B.png --a xx_A.png --out out/
====================================================
EOF
