"""大图重叠切块与概率图融合。

为什么必须重叠：车道线跨越瓦片边界时，两侧各自看到半条线，边缘必然断。
重叠 + 加权融合把接缝处的概率平滑掉。
若下游在矢量层合并跨瓦片结果，效果会更好（本工程只负责到概率图）。
"""

from __future__ import annotations

import numpy as np


def make_tiles(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """生成覆盖全图的重叠窗口 (y0, y1, x0, x1)。"""
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - tile + step), step))
    xs = list(range(0, max(1, w - tile + step), step))
    # 确保最后一块贴到边界
    if not ys or ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    if not xs or xs[-1] + tile < w:
        xs.append(max(0, w - tile))

    out = []
    for y0 in ys:
        for x0 in xs:
            out.append((y0, min(h, y0 + tile), x0, min(w, x0 + tile)))
    return out


def _window(tile: int, overlap: int, floor: float = 1e-2) -> np.ndarray:
    """二维余弦窗：中心权重 1，边缘渐隐，用于消除接缝。

    边缘保留一个很小的正权重（floor）而不是 0：图像最外圈的像素只被一个
    瓦片覆盖，权重为 0 会让归一化分母也为 0，该处结果直接变成 0（黑边）。
    只要权重非零，加权平均就精确等于那个瓦片的值。
    """
    ramp = np.ones(tile, dtype=np.float32)
    if overlap > 0:
        k = min(overlap, tile // 2)
        t = np.linspace(0, np.pi, k, dtype=np.float32)
        fade = 0.5 - 0.5 * np.cos(t)  # 0 -> 1
        fade = floor + (1.0 - floor) * fade
        ramp[:k] = fade
        ramp[-k:] = fade[::-1]
    return np.outer(ramp, ramp)


def fuse(probs: list[np.ndarray], boxes: list[tuple[int, int, int, int]], shape: tuple[int, int], overlap: int) -> np.ndarray:
    """按余弦窗加权融合各瓦片概率图。"""
    acc = np.zeros(shape, dtype=np.float32)
    wsum = np.zeros(shape, dtype=np.float32)
    for p, (y0, y1, x0, x1) in zip(probs, boxes):
        th, tw = y1 - y0, x1 - x0
        w = _window(max(th, tw), overlap)[:th, :tw]
        acc[y0:y1, x0:x1] += p[:th, :tw] * w
        wsum[y0:y1, x0:x1] += w
    return acc / np.maximum(wsum, 1e-6)


def pad_to_tile(img: np.ndarray, tile: int) -> tuple[np.ndarray, tuple[int, int]]:
    """把图补到 >= tile 的尺寸，返回 (padded, (h, w))。"""
    h, w = img.shape[:2]
    ph, pw = max(tile, h), max(tile, w)
    if (ph, pw) == (h, w):
        return img, (h, w)
    pad = [(0, ph - h), (0, pw - w)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad, mode="reflect"), (h, w)
