#!/usr/bin/env python
"""从目录生成配对清单（jsonl）。

目录约定（文件名 stem 即配对键）：
    data/B/  0001.png ...      影像
    data/A/  0001.png ...      粗提取
    data/GT/ 0001.png ...      标注

    python scripts/prepare_manifest.py --b data/B --a data/A --gt data/GT \
        --out data/train.jsonl --split-train 0.9 --seed 42

额外产出 data/val.jsonl。路径默认写成相对 --root 的形式，方便整个数据目录搬移。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def index_dir(d: Path) -> dict[str, Path]:
    if not d.is_dir():
        return {}
    return {f.stem: f for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in IMG_EXTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", required=True, help="影像 B 目录")
    ap.add_argument("--a", default=None, help="粗提取 A 目录")
    ap.add_argument("--gt", required=True, help="标注 C 目录")
    ap.add_argument("--out", required=True, help="输出 train 清单路径")
    ap.add_argument("--root", default=None, help="清单中路径的基准目录，默认取 B 的父目录")
    ap.add_argument("--val-out", default=None, help="val 清单路径，默认与 train 同目录的 val.jsonl")
    ap.add_argument("--split-train", type=float, default=1.0, help="训练集比例，<1 时按 key 随机切分")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--absolute", action="store_true", help="写绝对路径")
    args = ap.parse_args()

    b_dir = Path(args.b)
    a_dir = Path(args.a) if args.a else None
    gt_dir = Path(args.gt)
    root = Path(args.root) if args.root else b_dir.parent

    b_map = index_dir(b_dir)
    gt_map = index_dir(gt_dir)
    a_map = index_dir(a_dir) if a_dir else {}

    if not b_map:
        raise SystemExit(f"[ERROR] {b_dir} 下没有图片")
    if not gt_map:
        raise SystemExit(f"[ERROR] {gt_dir} 下没有图片")

    keys = sorted(set(b_map) & set(gt_map))
    missing_gt = sorted(set(b_map) - set(gt_map))
    if missing_gt:
        print(f"[WARN] {len(missing_gt)} 个影像没有 GT，已跳过: {missing_gt[:5]}")
    missing_b = sorted(set(gt_map) - set(b_map))
    if missing_b:
        print(f"[WARN] {len(missing_b)} 个 GT 没有影像，已跳过")
    if a_dir is not None:
        no_a = [k for k in keys if k not in a_map]
        if no_a:
            print(f"[WARN] {len(no_a)} 个样本没有 A，将写 null（训练时该通道全零）: {no_a[:5]}")
    if not keys:
        raise SystemExit("[ERROR] 影像与 GT 没有同名配对")

    def rel(p: Path) -> str:
        return str(p.resolve()) if args.absolute else str(Path(p).resolve().relative_to(root.resolve()) if p.resolve().is_relative_to(root.resolve()) else p.resolve())

    items = []
    for k in keys:
        items.append(
            {
                "name": k,
                "rgb": rel(b_map[k]),
                "a": rel(a_map[k]) if k in a_map else None,
                "gt": rel(gt_map[k]),
            }
        )

    random.Random(args.seed).shuffle(items)
    n_train = len(items) if args.split_train >= 1.0 else max(1, int(round(len(items) * args.split_train)))
    train_items, val_items = items[:n_train], items[n_train:]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[OK] train: {len(train_items)} -> {out}")

    if val_items:
        val_out = Path(args.val_out) if args.val_out else out.with_name("val.jsonl")
        with open(val_out, "w", encoding="utf-8") as f:
            for it in val_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[OK] val:   {len(val_items)} -> {val_out}")
    print(f"[INFO] 路径基准 root = {root.resolve()}，配置里 data.root 指向它或留 null")


if __name__ == "__main__":
    main()
