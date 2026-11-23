#!/usr/bin/env python
"""冒烟测试：用合成数据跑极少量步数，验证训练+推理+评估整条链路。

不需要真实数据。CPU 上约 1–3 分钟，GPU 上几十秒。

    python scripts/smoke_test.py
    python scripts/smoke_test.py --size 128 --n 6 --keep

任何一步失败都会以非零码退出，可直接放进 CI 或部署验收。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], step: str) -> None:
    print(f"\n{'=' * 70}\n[STEP] {step}\n{'=' * 70}")
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"\n[FAIL] {step} 返回码 {r.returncode}")
        sys.exit(r.returncode)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=128, help="合成瓦片边长")
    p.add_argument("--n", type=int, default=6, help="合成样本数")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--keep", action="store_true", help="保留临时目录以便检查产物")
    p.add_argument("--workdir", default=None, help="指定工作目录（默认用临时目录）")
    return p.parse_args()


def main():
    args = parse_args()
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="lane_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 工作目录: {work}")

    data = work / "data"
    run([sys.executable, "scripts/make_synthetic.py", "--out", str(data), "--n", str(args.n), "--size", str(args.size)], "生成合成数据")

    # 冒烟测试用极小的模型和步数，只验证链路
    cfg = work / "smoke.yaml"
    cfg.write_text(
        f"""
data:
  train_manifest: {data}/train.jsonl
  val_manifest: {data}/val.jsonl
  root: {data}
  num_workers: 0
  gt_dilate: 1
  hint_dropout: 0.15
  augment: true
  color_jitter: 0.15

model:
  encoder: resnet18
  channels: [rgb, a, skel, endpoint, dir, dist]
  pretrained: false          # 冒烟测试不下载权重
  out_stride: 2
  decoder_ch: 32
  dropout: 0.0

train:
  out_dir: {work}/run
  epochs: {args.epochs}
  batch_size: 2
  lr: 1.0e-3
  weight_decay: 1.0e-4
  pct_start: 0.1
  pos_weight: 5.0
  w_bce: 1.0
  w_dice: 1.0
  w_cldice: 1.0
  skel_iter: 3
  clip_grad: 1.0
  amp: false
  seed: 0
  device: {args.device}
  log_every: 1
  val_every: 1
  limit: 0

eval:
  threshold: 0.5
  apls_samples: 20
""",
        encoding="utf-8",
    )

    run([sys.executable, "-m", "lane_refine.train", "--config", str(cfg), "--out", str(work / "run")], "训练")

    best = work / "run" / "best.pt"
    if not best.exists():
        print(f"[FAIL] 没有生成 {best}")
        sys.exit(1)

    run(
        [
            sys.executable,
            "-m",
            "lane_refine.infer",
            "--ckpt",
            str(best),
            "--manifest",
            str(data / "val.jsonl"),
            "--root",
            str(data),
            "--out",
            str(work / "eval"),
            "--eval",
            "--device",
            args.device,
        ],
        "推理与评估",
    )

    metrics = work / "eval" / "metrics.json"
    if not metrics.exists():
        print(f"[FAIL] 没有生成 {metrics}")
        sys.exit(1)

    summary = json.loads(metrics.read_text())["summary"]
    print("\n评估摘要:")
    for k, v in summary.items():
        print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")

    required = {"iou", "f1", "skel_f1", "apls", "breaks"}
    missing = required - set(summary)
    if missing:
        print(f"[FAIL] metrics.json 缺少字段: {missing}")
        sys.exit(1)

    # 冒烟测试只验证"能跑通"，不验证精度；但 APLS 不该是 NaN
    if summary["apls"] != summary["apls"]:
        print("[FAIL] apls 是 NaN")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("[PASS] 冒烟测试通过：数据 -> 训练 -> 推理 -> 评估 全链路正常")
    print("=" * 70)
    print("注意：这里用的是合成数据 + 极小模型 + 1 epoch，指标没有参考价值。")

    if not args.keep and not args.workdir:
        shutil.rmtree(work, ignore_errors=True)
        print("临时目录已清理（--keep 可保留）")
    else:
        print(f"产物保留在 {work}")


if __name__ == "__main__":
    main()
