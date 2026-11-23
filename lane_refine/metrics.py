"""评估指标：像素级 + 拓扑级。

只报 IoU 会被误导：IoU 奖励画粗线，且对"一条线断成两段"几乎不敏感。
拓扑指标直接对应业务痛点，必须一起看。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .hint import skeletonize_mask


def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred/gt 均为二值 (H, W)。返回 precision/recall/f1/iou。"""
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    fn = float((~p & g).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou}


def _label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    structure = np.ones((3, 3), np.uint8) if connectivity == 8 else None
    lab, n = ndi.label(mask, structure=structure)
    return lab, int(n)


def count_breaks(pred: np.ndarray, min_len: int = 5) -> int:
    """断裂指标：骨架中长度 >= min_len 的连通分量个数减一。

    复杂度 O(分量数)，不依赖 GT，可以在无标注数据上直接监控。
    值越大说明线碎得越厉害。完全没有线时返回 0。
    """
    skel = skeletonize_mask(pred)
    if not skel.any():
        return 0
    lab, n = _label_components(skel, connectivity=8)
    lengths = ndi.sum(skel, lab, index=np.arange(1, n + 1))
    big = int((lengths >= min_len).sum())
    return max(0, big - 1)


def skeleton_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> dict:
    """骨架 F1（带容差），比像素 F1 更贴近"线画对了没有"。"""
    ps = skeletonize_mask(pred)
    gs = skeletonize_mask(gt)
    if not ps.any() or not gs.any():
        return {"skel_precision": 0.0, "skel_recall": 0.0, "skel_f1": 0.0}

    def _match(src: np.ndarray, dst: np.ndarray) -> float:
        if not src.any():
            return 0.0
        d = ndi.distance_transform_edt(~dst)
        hit = (src & (d <= tol)).sum()
        return float(hit) / float(src.sum())

    prec = _match(ps, gs)
    rec = _match(gs, ps)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"skel_precision": prec, "skel_recall": rec, "skel_f1": f1}


def _graph_from_skeleton(skel: np.ndarray):
    """骨架 -> 图。节点为骨架像素，边为 8 邻接。"""
    import networkx as nx

    ys, xs = np.nonzero(skel)
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    g = nx.Graph()
    g.add_nodes_from(range(len(idx)))
    for (y, x), i in idx.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = idx.get((y + dy, x + dx))
                if j is not None and j > i:
                    g.add_edge(i, j, weight=float(np.hypot(dy, dx)))
    return g, idx


def apls(pred: np.ndarray, gt: np.ndarray, n_samples: int = 200, seed: int = 0) -> float:
    """APLS-lite：在 GT 图上取随机起终点，比较最短路径长度与预测图上的路径长度。

    标准 APLS 需要对完整路网枚举 s-t 对；这里用随机采样近似，速度快、可批处理。
    返回 [0,1]，1 表示拓扑完全一致。任一方无图时返回 0。
    """
    ps = skeletonize_mask(pred)
    gs = skeletonize_mask(gt)
    if not ps.any() or not gs.any():
        return 0.0

    g_pred, _ = _graph_from_skeleton(ps)

    # GT 侧先压缩成"最短路径距离"，避免在数万节点的图上反复 Dijkstra
    rng = np.random.default_rng(seed)
    gs_pts = np.argwhere(gs)
    if len(gs_pts) < 2:
        return 0.0
    # 只从度数 >= 2 的骨架上取点（端点路径太短，统计意义弱）
    deg = ndi.convolve(gs.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - gs
    cand = gs & (deg >= 2)
    cand_pts = np.argwhere(cand) if cand.sum() >= 2 else gs_pts

    n_pairs = min(n_samples, len(cand_pts) // 2)
    if n_pairs < 1:
        # 采样点不足说明 GT 骨架退化成极少数像素（例如整块掩膜全为 1），
        # 此时 APLS 无定义。返回 nan 而不是 0，避免被平均成"模型很差"。
        return float("nan")

    # 每次取两个相距较远的点
    ratios = []
    tries = 0
    while len(ratios) < n_pairs and tries < n_pairs * 5:
        tries += 1
        i, j = rng.choice(len(cand_pts), size=2, replace=False)
        p, q = cand_pts[i], cand_pts[j]
        dist = float(np.hypot(*(p - q)))
        if dist < 20:
            continue

        gt_path = _shortest_on_mask(gs, p, q)
        if not np.isfinite(gt_path) or gt_path <= 0:
            continue
        pred_path = _shortest_on_mask_for_pred(g_pred, ps, p, q)
        if not np.isfinite(pred_path):
            ratios.append(0.0)
            continue
        r = min(gt_path, pred_path) / max(gt_path, pred_path)
        ratios.append(r)

    # 一个有效采样对都没凑出来（骨架退化成一簇像素），APLS 无定义。
    # 返回 nan 而不是 0，避免被平均成"模型很差"。
    return float(np.mean(ratios)) if ratios else float("nan")


def _shortest_on_mask(mask: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """在掩膜上求 p->q 的最短 8 连通路径长度；不连通返回 inf。"""
    import heapq

    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    pts = set(zip(ys.tolist(), xs.tolist()))
    start = (int(p[0]), int(p[1]))
    goal = (int(q[0]), int(q[1]))
    if start not in pts or goal not in pts:
        return float("inf")

    dist = {start: 0.0}
    pq = [(0.0, start)]
    while pq:
        d, cur = heapq.heappop(pq)
        if cur == goal:
            return d
        if d > dist.get(cur, float("inf")):
            continue
        cy, cx = cur
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (cy + dy, cx + dx)
                if nb in pts:
                    nd = d + float(np.hypot(dy, dx))
                    if nd < dist.get(nb, float("inf")):
                        dist[nb] = nd
                        heapq.heappush(pq, (nd, nb))
    return float("inf")


def _shortest_on_mask_for_pred(g, mask: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """预测图上取离 p/q 最近的骨架节点再求最短路。"""
    import networkx as nx

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return float("inf")
    pts = np.stack([ys, xs], axis=1)
    i = int(np.argmin(((pts - p) ** 2).sum(1)))
    j = int(np.argmin(((pts - q) ** 2).sum(1)))
    # 骨架节点在图中按 nonzero 顺序编号，索引与 pts 一一对应
    try:
        return float(nx.shortest_path_length(g, i, j, weight="weight"))
    except Exception:
        return float("inf")


def evaluate(pred: np.ndarray, gt: np.ndarray, compute_apls: bool = True, n_apls: int = 200, seed: int = 0) -> dict:
    """一次性算完所有指标。"""
    m = pixel_metrics(pred, gt)
    m.update(skeleton_f1(pred, gt))
    m["breaks"] = count_breaks(pred)
    m["gt_breaks"] = count_breaks(gt)
    m["break_excess"] = m["breaks"] - m["gt_breaks"]
    if compute_apls:
        m["apls"] = apls(pred, gt, n_samples=n_apls, seed=seed)
    return m
