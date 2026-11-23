"""从粗提取车道线 A 派生条件通道。

派生内容
    skeleton  细化为单像素宽骨架（去除 A 的线宽差异）
    endpoint  A 矢量化后的端点，直接标记"这里需要连接"
    dir       方向场，用双角编码 (cos2θ, sin2θ) 表示，对线方向 180° 周期天然无歧义
    dist      到最近车道线像素的距离场，帮助网络判断"离道路有多远"

端点与方向场分别对应两类主要错误：断裂（缺连接）与合流（方向消歧）。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except ImportError:  # pragma: no cover
    _sk_skeletonize = None

_NEIGHBOR_KERNEL = np.ones((3, 3), dtype=np.uint8)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """二值掩膜 -> 单像素骨架。mask 支持 bool 或 0/1/255 数值。"""
    binary = np.asarray(mask) > 0.5 if mask.dtype != bool else mask
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    if _sk_skeletonize is not None:
        return _sk_skeletonize(binary)
    return _thinning_fallback(binary)


def _thinning_fallback(binary: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """skimage 缺失时的形态学细化（Zhang-Suen 简化版，仅用于兜底）。"""
    img = binary.astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            nb = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            trans = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (img == 1) & (nb >= 2) & (nb <= 6) & (trans == 1) & cond
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def find_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """骨架端点：8 邻域内恰好 1 个邻居的像素。"""
    skel = skeleton.astype(np.uint8)
    if not skel.any():
        return np.zeros_like(skel, dtype=bool)
    n = ndi.convolve(skel, _NEIGHBOR_KERNEL, mode="constant", cval=0) - skel
    return (skel == 1) & (n == 1)


def direction_field(mask: np.ndarray, sigma: float = 2.0, near_px: int = 3) -> np.ndarray:
    """结构张量估计线方向，返回 (2, H, W) 的双角编码场 (cos2θ, sin2θ)。

    对细线而言梯度垂直于线，故线方向 = 主梯度方向 + 90°。
    双角编码让 θ 与 θ+180° 等价，避免方向符号歧义。
    """
    m = mask.astype(np.float32)
    h, w = m.shape
    out = np.zeros((2, h, w), dtype=np.float32)
    if not m.any():
        return out

    blur = ndi.gaussian_filter(m, sigma)
    ix = ndi.sobel(blur, axis=1, mode="nearest")
    iy = ndi.sobel(blur, axis=0, mode="nearest")

    jxx = ndi.gaussian_filter(ix * ix, sigma)
    jyy = ndi.gaussian_filter(iy * iy, sigma)
    jxy = ndi.gaussian_filter(ix * iy, sigma)

    theta_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    # 线方向 = 梯度方向 + π/2 → 双角加 π → 整体取负
    out[0] = -np.cos(2.0 * theta_grad)
    out[1] = -np.sin(2.0 * theta_grad)

    near = ndi.binary_dilation(mask, iterations=near_px)
    out[:, ~near] = 0.0

    # 各向同性处（无明确方向）置零，避免给网络注入噪声方向
    aniso = np.abs(jxx - jyy) + 2.0 * np.abs(jxy)
    weak = aniso < (0.05 * (np.abs(jxx) + np.abs(jyy) + 1e-6))
    out[:, weak & ~near] = 0.0
    return out


def distance_field(mask: np.ndarray, max_dist: float = 24.0) -> np.ndarray:
    """到最近车道线像素的欧氏距离，除以 max_dist 后裁剪到 [0,1]。"""
    if not mask.any():
        return np.ones_like(mask, dtype=np.float32)
    d = ndi.distance_transform_edt(~mask.astype(bool)).astype(np.float32)
    return np.clip(d / max_dist, 0.0, 1.0)


def compute_hints(mask: np.ndarray, sigma: float = 2.0, near_px: int = 3, max_dist: float = 24.0) -> dict:
    """一次性算出全部派生通道。mask: (H, W) 0/1。"""
    binary = np.asarray(mask) > 0.5
    skel = skeletonize_mask(binary)
    return {
        "skeleton": skel.astype(np.float32),
        "endpoint": find_endpoints(skel).astype(np.float32),
        "dir": direction_field(binary, sigma=sigma, near_px=near_px),
        "dist": distance_field(binary, max_dist=max_dist),
    }


CHANNEL_BUILDERS = ("rgb", "a", "skel", "endpoint", "dir", "dist")

# 每个逻辑通道占用的通道数
CHANNEL_WIDTH = {"rgb": 3, "a": 1, "skel": 1, "endpoint": 1, "dir": 2, "dist": 1}


def num_channels(channels: list[str]) -> int:
    unknown = [c for c in channels if c not in CHANNEL_WIDTH]
    if unknown:
        raise ValueError(f"未知通道 {unknown}，可选: {list(CHANNEL_WIDTH)}")
    return sum(CHANNEL_WIDTH[c] for c in channels)
