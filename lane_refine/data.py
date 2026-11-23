"""数据集：读取 (B 影像, A 粗提取, GT 车道线)，组装多通道输入。

设计要点
    - GT 训练时膨胀 1 px：细线任务里这能显著稳定早期训练，推理后再骨架化回 1 px
    - Hint Dropout：整张 A 派生通道随机置零，防止网络无脑复制 A 的错误
    - 增强只用 8 种二面体变换（翻转 + 90° 旋转）+ 可选亮度/对比度抖动，
      不用随机裁剪/缩放，保证线宽语义不变
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .hint import compute_hints
from .imops import IMAGENET_MEAN, IMAGENET_STD, dilate, jitter, load_mask, load_rgb, resize_mask


def _dihedral(img: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """8 种二面体变换之一。img 可以是 (H,W,C) 或 (H,W)。"""
    if flip:
        img = np.flip(img, axis=1)
    if k:
        img = np.rot90(img, k=k, axes=(0, 1))
    return img


class LaneTileDataset(Dataset):
    """瓦片级数据集。items 来自清单，每项含 rgb / a / gt 三个路径。"""

    def __init__(
        self,
        items: list[dict],
        channels: list[str],
        train: bool = True,
        gt_dilate: int = 1,
        hint_dropout: float = 0.15,
        augment: bool = True,
        color_jitter: float = 0.15,
    ):
        self.items = items
        self.channels = channels
        self.train = train
        self.gt_dilate = int(gt_dilate)
        self.hint_dropout = float(hint_dropout)
        self.augment = bool(augment and train)
        self.color_jitter = float(color_jitter)

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        it = self.items[idx]
        rgb = load_rgb(it["rgb"])
        a = load_mask(it["a"]) if it.get("a") else np.zeros(rgb.shape[:2], dtype=np.float32)
        gt = load_mask(it["gt"]) if it.get("gt") else np.zeros(rgb.shape[:2], dtype=np.float32)

        if a.shape[:2] != rgb.shape[:2]:
            a = resize_mask(a, rgb.shape[:2])
        if gt.shape[:2] != rgb.shape[:2]:
            gt = resize_mask(gt, rgb.shape[:2])
        return rgb, a, gt

    def _build_input(self, rgb: np.ndarray, a: np.ndarray, drop_hints: bool) -> np.ndarray:
        h, w = rgb.shape[:2]
        zero1 = np.zeros((h, w, 1), np.float32)
        zero2 = np.zeros((h, w, 2), np.float32)

        parts = []
        hints = None
        for ch in self.channels:
            if ch == "rgb":
                parts.append((rgb - IMAGENET_MEAN) / IMAGENET_STD)
                continue
            if ch not in ("a", "skel", "endpoint", "dir", "dist"):
                raise ValueError(f"未知通道 {ch}")
            if drop_hints:
                parts.append(zero2 if ch == "dir" else zero1)
                continue
            if ch == "a":
                parts.append(a[..., None])
                continue
            if hints is None:
                hints = compute_hints(a)
            if ch == "dir":
                parts.append(np.moveaxis(hints["dir"], 0, -1))
            else:
                key = {"skel": "skeleton"}.get(ch, ch)
                parts.append(hints[key][..., None])
        return np.concatenate(parts, axis=-1)

    def __getitem__(self, idx: int):
        rgb, a, gt = self._load(idx)

        if self.augment:
            k = random.randint(0, 3)
            flip = random.random() < 0.5
            rgb = _dihedral(rgb, k, flip)
            a = _dihedral(a, k, flip)
            gt = _dihedral(gt, k, flip)
            if self.color_jitter > 0 and random.random() < 0.5:
                rgb = jitter(rgb, self.color_jitter, random)

        if self.gt_dilate > 0:
            gt = dilate(gt, self.gt_dilate)

        drop_hints = self.train and self.hint_dropout > 0 and random.random() < self.hint_dropout
        x = self._build_input(rgb, a, drop_hints)

        x = np.ascontiguousarray(x.transpose(2, 0, 1))  # HWC -> CHW
        y = gt[None]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)
