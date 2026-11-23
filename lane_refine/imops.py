"""与 torch 无关的图像/掩膜基础操作。

单独抽出，让推理、评估、矢量化这些只在 CPU 上跑的逻辑不必 import torch。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path: str) -> np.ndarray:
    """读 RGB，返回 float32 [0,1] 的 (H, W, 3)。"""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path: str) -> np.ndarray:
    """读掩膜。大于 0 即视为前景，兼容 0/1、0/255、单通道。"""
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > 0).astype(np.float32)


def resize_mask(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """最近邻缩放掩膜到 (H, W)，保持二值。"""
    img = Image.fromarray((m * 255).astype(np.uint8))
    img = img.resize((size[1], size[0]), Image.NEAREST)
    return (np.asarray(img) > 127).astype(np.float32)


def dilate(m: np.ndarray, iters: int) -> np.ndarray:
    from scipy import ndimage as ndi

    if iters <= 0:
        return (m > 0.5).astype(np.float32)
    return ndi.binary_dilation(m > 0.5, iterations=iters).astype(np.float32)


def binarize(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """概率图 -> 二值掩膜（float32, 0/1）。后处理交给下游。"""
    return (prob > threshold).astype(np.float32)


def save_mask(m: np.ndarray, path) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((m > 0.5).astype(np.uint8) * 255).save(p)


def jitter(rgb: np.ndarray, strength: float, rng) -> np.ndarray:
    """亮度/对比度随机抖动。rng 需提供 random() 与 uniform(a,b)。"""
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    f = 1.0 + rng.uniform(-strength, strength)
    img = ImageEnhance.Brightness(img).enhance(f) if rng.random() < 0.5 else ImageEnhance.Contrast(img).enhance(f)
    return np.asarray(img, dtype=np.float32) / 255.0
