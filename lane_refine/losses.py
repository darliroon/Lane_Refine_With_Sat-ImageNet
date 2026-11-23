"""损失函数：加权 BCE + Dice + clDice。

clDice(centerline Dice, Shit et al. CVPR2021) 直接优化骨架连通性，
对"一条长线断成两段"的惩罚远大于同区域的 IoU 损失，是断裂问题的主要解药。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    p3 = -F.max_pool2d(-img, (3, 3), (1, 1), (1, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skel(img: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
    """可微骨架近似：反复开运算取差值并累积。"""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(n_iter):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cl_dice_loss(logits: torch.Tensor, target: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    """1 - clDice。logits/target: (N, 1, H, W)。"""
    pred = torch.sigmoid(logits)
    skel_pred = soft_skel(pred, n_iter)
    skel_true = soft_skel(target, n_iter)

    t_prec = (skel_pred * target).sum(dim=(1, 2, 3)) / (skel_pred.sum(dim=(1, 2, 3)) + 1e-6)
    t_sens = (skel_true * pred).sum(dim=(1, 2, 3)) / (skel_true.sum(dim=(1, 2, 3)) + 1e-6)
    cl = 2.0 * t_prec * t_sens / (t_prec + t_sens + 1e-6)
    return 1.0 - cl.mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    num = 2.0 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return (1.0 - num / den).mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 5.0) -> torch.Tensor:
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


class CombinedLoss(torch.nn.Module):
    """BCE + Dice + clDice 加权和。"""

    def __init__(self, pos_weight: float = 5.0, w_bce: float = 1.0, w_dice: float = 1.0, w_cldice: float = 1.0, skel_iter: int = 10):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)
        self.w_cldice = float(w_cldice)
        self.skel_iter = int(skel_iter)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        parts = {}
        total = logits.new_zeros(())
        if self.w_bce > 0:
            parts["bce"] = bce_loss(logits, target, self.pos_weight)
            total = total + self.w_bce * parts["bce"]
        if self.w_dice > 0:
            parts["dice"] = dice_loss(logits, target)
            total = total + self.w_dice * parts["dice"]
        if self.w_cldice > 0:
            parts["cldice"] = cl_dice_loss(logits, target, self.skel_iter)
            total = total + self.w_cldice * parts["cldice"]
        return total, {k: float(v.detach()) for k, v in parts.items()}
