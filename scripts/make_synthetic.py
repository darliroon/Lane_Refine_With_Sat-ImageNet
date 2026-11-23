#!/usr/bin/env python
"""生成合成数据，用于在真实数据到位前验证整条流水线能跑通。

每张样本由程序绘制：
    GT   多段车道线（含分叉、合流、弯道），线宽 2px
    A    从 GT 派生的"劣化版"：随机断开、生成短毛刺、在合流处错位
    B    在 GT 周围绘制路面底纹（模拟真实影像的纹理），带噪声

这组数据只用来验证代码路径正确、损失能下降、修复能起作用，
真实效果必须在你的数据上评估。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _smooth_curve(rng, x0, y0, x1, y1, bend, n=60):
    """二次贝塞尔采样，返回 (x, y) 点列。"""
    cx = (x0 + x1) / 2 + rng.uniform(-bend, bend)
    cy = (y0 + y1) / 2 + rng.uniform(-bend, bend)
    ts = np.linspace(0, 1, n)
    xs = (1 - ts) ** 2 * x0 + 2 * (1 - ts) * ts * cx + ts**2 * x1
    ys = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * cy + ts**2 * y1
    return np.stack([xs, ys], axis=1)


def draw_gt(rng, size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    canvas = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(canvas)
    curves = []
    n_main = rng.integers(2, 4)

    for _ in range(n_main):
        side = rng.integers(0, 4)
        if side == 0:
            p0 = (rng.uniform(0, size * 0.2), rng.uniform(0, size))
            p1 = (rng.uniform(size * 0.7, size), rng.uniform(0, size))
        elif side == 1:
            p0 = (rng.uniform(0, size), rng.uniform(0, size * 0.2))
            p1 = (rng.uniform(0, size), rng.uniform(size * 0.7, size))
        else:
            p0 = (rng.uniform(size * 0.1, size * 0.4), rng.uniform(0, size))
            p1 = (rng.uniform(size * 0.6, size * 0.9), rng.uniform(0, size))
        c = _smooth_curve(rng, p0[0], p0[1], p1[0], p1[1], bend=size * 0.25)
        curves.append(c)
        d.line([tuple(p) for p in c], fill=255, width=2)

    # 分叉：从主线上某点引出支线
    for _ in range(rng.integers(1, 3)):
        src = curves[rng.integers(0, len(curves))]
        idx = rng.integers(len(src) // 4, 3 * len(src) // 4)
        x, y = src[idx]
        ang = rng.uniform(0, 2 * math.pi)
        L = rng.uniform(size * 0.15, size * 0.35)
        ex, ey = x + L * math.cos(ang), y + L * math.sin(ang)
        c = _smooth_curve(rng, x, y, ex, ey, bend=size * 0.15, n=30)
        curves.append(c)
        d.line([tuple(p) for p in c], fill=255, width=2)

    return np.asarray(canvas) > 0, curves


def degrade_to_a(rng, gt: np.ndarray, curves: list[np.ndarray]) -> np.ndarray:
    """模拟"粗提取"：断线 + 毛刺。"""
    a = gt.copy()
    for c in curves:
        # 随机挖掉若干小段模拟断裂
        for _ in range(rng.integers(1, 3)):
            if len(c) < 12:
                continue
            i = rng.integers(3, len(c) - 3)
            j = min(len(c) - 1, i + rng.integers(3, 9))
            for p in c[i:j]:
                y, x = int(p[1]), int(p[0])
                a[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = False
        # 毛刺
        if rng.random() < 0.6:
            i = rng.integers(0, len(c))
            x, y = c[i]
            ang = rng.uniform(0, 2 * math.pi)
            L = rng.uniform(4, 14)
            ys = np.linspace(y, y + L * math.sin(ang), int(L) + 2).astype(int).clip(0, a.shape[0] - 1)
            xs = np.linspace(x, x + L * math.cos(ang), int(L) + 2).astype(int).clip(0, a.shape[1] - 1)
            a[ys, xs] = True
    return a


def draw_imagery(rng, gt: np.ndarray, size: int) -> np.ndarray:
    """用 GT 作"路面"骨架，画出灰度底纹，模拟卫片。"""
    base = rng.normal(0.55, 0.06, (size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    base += 0.05 * np.sin(xx / rng.uniform(10, 25)) * np.cos(yy / rng.uniform(10, 25))

    road = Image.fromarray((gt * 255).astype(np.uint8))
    road = road.filter(__import__("PIL.ImageFilter", fromlist=["ImageFilter"]).GaussianBlur(rng.uniform(2, 5)))
    road = np.asarray(road, dtype=np.float32) / 255.0
    img = base * (1 - road) + (base * 0.35 + 0.5) * road
    img = np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1)
    return (np.stack([img, img, img], -1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出数据根目录")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-train", type=float, default=0.8)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    for sub in ("B", "A", "GT"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    items = []
    for i in range(args.n):
        gt, curves = draw_gt(rng, args.size)
        a = degrade_to_a(rng, gt, curves)
        b = draw_imagery(rng, gt, args.size)

        name = f"{i:04d}"
        Image.fromarray(b).save(root / "B" / f"{name}.png")
        Image.fromarray((a * 255).astype(np.uint8)).save(root / "A" / f"{name}.png")
        Image.fromarray((gt * 255).astype(np.uint8)).save(root / "GT" / f"{name}.png")
        items.append({"name": name, "rgb": f"B/{name}.png", "a": f"A/{name}.png", "gt": f"GT/{name}.png"})

    rng.shuffle(items)
    n_train = max(1, int(round(len(items) * args.split_train)))
    for split, subset in (("train", items[:n_train]), ("val", items[n_train:])):
        p = root / f"{split}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for it in subset:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[OK] {split}: {len(subset)} -> {p}")

    print(f"[OK] 合成数据在 {root.resolve()}")
    print(f"     训练: python -m lane_refine.train --config configs/base.yaml --manifest {root}/train.jsonl --val_manifest {root}/val.jsonl --out runs/synth")


if __name__ == "__main__":
    main()
