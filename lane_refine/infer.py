#!/usr/bin/env python
"""推理：给定 B 影像与 A 粗提取，输出精化后的 C 掩膜与概率图。

    # 单张
    python -m lane_refine.infer --ckpt runs/base/best.pt \
        --rgb tile_B.png --a tile_A.png --out out/

    # 批量（清单）
    python -m lane_refine.infer --ckpt runs/base/best.pt \
        --manifest data/test.jsonl --out runs/eval --eval

    # 大图（自动重叠切块）
    python -m lane_refine.infer --ckpt runs/base/best.pt \
        --rgb big_B.png --a big_A.png --out out/ --tile 512 --overlap 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .imops import IMAGENET_MEAN, IMAGENET_STD, binarize, load_mask, load_rgb, resize_mask, save_mask
from .hint import compute_hints
from .metrics import evaluate
from .model import build_model
from .tiling import fuse, make_tiles, pad_to_tile
from .utils import get_device, read_manifest, resolve_paths, setup_logging, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Lane refinement inference")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rgb", default=None, help="单张影像 B")
    p.add_argument("--a", default=None, help="单张粗提取 A（可缺省，缺省时条件通道置零）")
    p.add_argument("--gt", default=None, help="单张 GT C（可选，给了就算指标）")
    p.add_argument("--manifest", default=None, help="批量清单 jsonl/csv")
    p.add_argument("--root", default=None, help="清单中相对路径的基准目录")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=None, help="覆盖配置中的二值化阈值")
    p.add_argument("--tile", type=int, default=0, help=">0 启用重叠切块推理")
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--save_prob", action="store_true", help="保存 float16 概率图 .npy，供下游后处理使用")
    p.add_argument("--eval", action="store_true", help="计算指标并写 metrics.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def build_input(rgb: np.ndarray, a: np.ndarray, channels: list[str]) -> np.ndarray:
    h, w = rgb.shape[:2]
    parts, hints = [], None
    for ch in channels:
        if ch == "rgb":
            parts.append((rgb - IMAGENET_MEAN) / IMAGENET_STD)
        elif ch == "a":
            parts.append(a[..., None])
        else:
            if hints is None:
                hints = compute_hints(a)
            if ch == "skel":
                parts.append(hints["skeleton"][..., None])
            elif ch == "endpoint":
                parts.append(hints["endpoint"][..., None])
            elif ch == "dir":
                parts.append(np.moveaxis(hints["dir"], 0, -1))
            elif ch == "dist":
                parts.append(hints["dist"][..., None])
            else:
                raise ValueError(f"未知通道 {ch}")
    return np.concatenate(parts, axis=-1)


@torch.no_grad()
def predict_array(model, rgb: np.ndarray, a: np.ndarray, channels: list[str], device, tile: int = 0, overlap: int = 128) -> np.ndarray:
    """返回 (H, W) 概率图。"""
    if rgb.shape[:2] != a.shape[:2]:
        a = resize_mask(a, rgb.shape[:2])

    if tile <= 0:
        x = build_input(rgb, a, channels).transpose(2, 0, 1)[None]
        logits = model(torch.from_numpy(np.ascontiguousarray(x)).float().to(device))
        import torch.nn.functional as F

        if logits.shape[-2:] != rgb.shape[:2]:
            logits = F.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        return torch.sigmoid(logits.float())[0, 0].cpu().numpy()

    h, w = rgb.shape[:2]
    rgb_p, _ = pad_to_tile(rgb, tile)
    a_p, _ = pad_to_tile(a, tile)
    hp, wp = rgb_p.shape[:2]
    boxes = make_tiles(hp, wp, tile, overlap)

    probs = []
    for y0, y1, x0, x1 in boxes:
        sub_rgb = rgb_p[y0:y1, x0:x1]
        sub_a = a_p[y0:y1, x0:x1]
        x = build_input(sub_rgb, sub_a, channels).transpose(2, 0, 1)[None]
        logits = model(torch.from_numpy(np.ascontiguousarray(x)).float().to(device))
        probs.append(torch.sigmoid(logits.float())[0, 0].cpu().numpy())

    fused = fuse(probs, boxes, (hp, wp), overlap)
    return fused[:h, :w]


def process_one(model, item: dict, channels: list[str], device, args, out_dir: Path, log) -> dict:
    name = item.get("name") or Path(item["rgb"]).stem
    rgb = load_rgb(item["rgb"])
    a = load_mask(item["a"]) if item.get("a") else np.zeros(rgb.shape[:2], np.float32)

    prob = predict_array(model, rgb, a, channels, device, tile=args.tile, overlap=args.overlap)
    if args.save_prob:
        np.save(out_dir / f"{name}_prob.npy", prob.astype(np.float16))

    mask = binarize(prob, args.threshold)
    result = {"name": name}
    save_mask(mask, out_dir / f"{name}_C.png")

    if item.get("gt"):
        gt = load_mask(item["gt"])
        if gt.shape != mask.shape:
            gt = resize_mask(gt, mask.shape)
        result.update(evaluate(mask, gt, compute_apls=args.eval, n_apls=100))
    return result


def main():
    args = parse_args()
    log = setup_logging()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    channels = cfg["model"]["channels"]
    device = get_device(args.device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    log.info("加载 %s (epoch=%s)", args.ckpt, ck.get("epoch"))

    if args.threshold is None:
        args.threshold = cfg["eval"].get("threshold", 0.5)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rgb:
        items = [{"name": Path(args.rgb).stem, "rgb": args.rgb, "a": args.a, "gt": args.gt}]
    elif args.manifest:
        items = resolve_paths(read_manifest(args.manifest), base=args.root or cfg["data"].get("root"))
    else:
        raise SystemExit("需要 --rgb 或 --manifest")

    if args.limit > 0:
        items = items[: args.limit]

    results = []
    for item in items:
        r = process_one(model, item, channels, device, args, out_dir, log)
        log.info("%s: %s", r["name"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "name"})
        results.append(r)

    if args.eval and items and "iou" in results[0]:
        keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
        summary = {k: float(np.nanmean([r[k] for r in results if k in r])) for k in keys}
        write_json(out_dir / "metrics.json", {"summary": summary, "per_tile": results})
        log.info("平均: %s", {k: round(v, 4) for k, v in summary.items()})
    elif len(results) > 1:
        write_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
