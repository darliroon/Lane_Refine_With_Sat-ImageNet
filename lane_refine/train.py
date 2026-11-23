#!/usr/bin/env python
"""训练入口。

    python -m lane_refine.train --config configs/base.yaml --manifest data/train.jsonl

支持的配置见 configs/base.yaml；命令行参数覆盖配置文件。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import LaneTileDataset, collate
from .hint import num_channels
from .losses import CombinedLoss
from .metrics import evaluate
from .model import build_model
from .utils import get_device, read_manifest, resolve_paths, set_seed, setup_logging, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Train lane refinement segmentation model")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--manifest", default=None, help="训练清单（覆盖配置）")
    p.add_argument("--val_manifest", default=None)
    p.add_argument("--out", default=None, help="输出目录（覆盖配置）")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", default=None, help="从 checkpoint 恢复")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0, help="调试：只用前 N 个样本")
    return p.parse_args()


def load_cfg(args) -> dict:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.manifest:
        cfg["data"]["train_manifest"] = args.manifest
    if args.val_manifest:
        cfg["data"]["val_manifest"] = args.val_manifest
    if args.out:
        cfg["train"]["out_dir"] = args.out
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.device:
        cfg["train"]["device"] = args.device
    return cfg


def build_loader(cfg, split: str, log, distributed: bool = False):
    d = cfg["data"]
    key = f"{split}_manifest"
    manifest = d.get(key)
    if not manifest:
        return None
    items = resolve_paths(read_manifest(manifest), base=d.get("root"))
    if cfg["train"].get("limit"):
        items = items[: cfg["train"]["limit"]]
    if not items:
        raise ValueError(f"{key} 为空: {manifest}")

    ds = LaneTileDataset(
        items,
        channels=cfg["model"]["channels"],
        train=(split == "train"),
        # gt_dilate 是训练技巧（让细线更容易被优化），绝不能用在验证集上，
        # 否则指标会虚高（实测 IoU 0.502 vs 真实 0.265）。
        gt_dilate=d.get("gt_dilate", 1) if split == "train" else 0,
        hint_dropout=d.get("hint_dropout", 0.15) if split == "train" else 0.0,
        augment=d.get("augment", True) if split == "train" else False,
        color_jitter=d.get("color_jitter", 0.15),
    )
    log.info("%s: %d 样本, 输入通道 %d", split, len(ds), num_channels(cfg["model"]["channels"]))

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds, shuffle=(split == "train"), drop_last=(split == "train"))

    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=d.get("num_workers", 4),
        collate_fn=collate,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=d.get("num_workers", 4) > 0,
    )


def infer_prob(model, x: torch.Tensor, out_hw: tuple[int, int] | None = None, amp: bool = True) -> torch.Tensor:
    """前向并还原到原分辨率，返回 (N,1,H,W) 概率。"""
    with torch.autocast(device_type=x.device.type, enabled=amp and x.device.type == "cuda"):
        logits = model(x)
    if logits.shape[-2:] != x.shape[-2:]:
        import torch.nn.functional as F

        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return torch.sigmoid(logits.float())


@torch.no_grad()
def validate(model, loader, cfg, log, compute_apls=True, distributed=False) -> dict:
    model.eval()
    thresh = cfg["eval"].get("threshold", 0.5)
    agg = []
    dev = next(model.parameters()).device
    for x, y in loader:
        x = x.to(dev)
        prob = infer_prob(model, x)
        pred = (prob > thresh).cpu().numpy()[:, 0]
        gt = (y.numpy()[:, 0] > 0.5)
        for p, g in zip(pred, gt):
            agg.append(evaluate(p, g, compute_apls=compute_apls, n_apls=cfg["eval"].get("apls_samples", 100)))

    if not agg:
        return {}

    if distributed:
        # 各进程只看到部分验证样本，汇总后再算均值
        local = torch.tensor([np.mean([a[k] for a in agg]) for k in agg[0]], dtype=torch.float64, device=dev)
        cnt = torch.tensor([float(len(agg))], dtype=torch.float64, device=dev)
        gathered = [torch.zeros_like(local) for _ in range(torch.distributed.get_world_size())]
        counts = [torch.zeros_like(cnt) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, local)
        torch.distributed.all_gather(counts, cnt)
        total = torch.stack(counts).sum()
        merged = (torch.stack(gathered) * torch.stack(counts)).sum(0) / total.clamp(min=1)
        return {k: float(v) for k, v in zip(agg[0].keys(), merged.cpu())}

    # apls 在退化的 GT 上返回 nan，用 nanmean 跳过这些样本
    return {k: float(np.nanmean([a[k] for a in agg])) for k in agg[0]}


def _make_scheduler(opt, t: dict, total_steps: int):
    """学习率调度器。

    OneCycleLR 在 total_steps 很小（< ~10）时，线性退火段会越过 0 产生负学习率，
    甚至触发 ZeroDivisionError。总步数太少时没有调度的意义，直接恒定学习率。
    """
    total_steps = max(1, int(total_steps))
    if total_steps < 10:
        return torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=total_steps)

    pct_start = float(t.get("pct_start", 0.1))
    # pct_start 过小会让 warmup 段不足一步，同样会算出发散的学习率
    pct_start = max(pct_start, 2.0 / total_steps)
    return torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=t["lr"],
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="linear",
    )


def _make_scaler(device, amp: bool):
    """混合精度的 GradScaler，兼容 torch 2.1–2.3 的旧签名。"""
    enabled = bool(amp and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def main():
    args = parse_args()
    cfg = load_cfg(args)
    log = setup_logging()

    # ---- 分布式初始化（torchrun 启动时环境变量由 torchrun 注入）----
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
    is_main = (not distributed) or torch.distributed.get_rank() == 0

    out_dir = Path(cfg["train"]["out_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

    set_seed(cfg["train"].get("seed", 42) + local_rank)
    device = get_device(cfg["train"].get("device", "cuda"))
    if distributed:
        device = torch.device("cuda", local_rank)
    log.info("rank=%s device=%s", local_rank, device)

    cfg["model"]["in_channels"] = num_channels(cfg["model"]["channels"])
    model = build_model(cfg["model"]).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    n_par = sum(p.numel() for p in model.parameters())
    if is_main:
        log.info("模型 %s, 参数 %.1fM, 输入通道 %d", cfg["model"].get("encoder"), n_par / 1e6, cfg["model"]["in_channels"])

    train_loader = build_loader(cfg, "train", log, distributed=distributed)
    val_loader = build_loader(cfg, "val", log, distributed=distributed)

    t = cfg["train"]
    crit = CombinedLoss(
        pos_weight=t.get("pos_weight", 5.0),
        w_bce=t.get("w_bce", 1.0),
        w_dice=t.get("w_dice", 1.0),
        w_cldice=t.get("w_cldice", 1.0),
        skel_iter=t.get("skel_iter", 10),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
    steps_per_epoch = max(1, len(train_loader))
    sched = _make_scheduler(opt, t, t["epochs"] * steps_per_epoch)

    scaler = _make_scaler(device, t.get("amp", True))

    start_epoch = 0
    best = -1.0
    history = []
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck.get("epoch", 0)
        best = ck.get("best", -1.0)
        log.info("从 %s 恢复，epoch=%d best=%.4f", args.resume, start_epoch, best)

    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        t0 = time.time()
        run_total, run_parts, n_batch = 0.0, {}, 0

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(x)
                if logits.shape[-2:] != y.shape[-2:]:
                    import torch.nn.functional as F

                    logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss, parts = crit(logits.float(), y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("clip_grad", 1.0))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("clip_grad", 1.0))
                opt.step()
            sched.step()

            run_total += float(loss.detach())
            for k, v in parts.items():
                run_parts[k] = run_parts.get(k, 0.0) + v
            n_batch += 1

            if (step + 1) % t.get("log_every", 20) == 0:
                log.info(
                    "epoch %d [%d/%d] loss=%.4f %s lr=%.2e",
                    epoch,
                    step + 1,
                    steps_per_epoch,
                    run_total / n_batch,
                    " ".join(f"{k}={v / n_batch:.4f}" for k, v in run_parts.items()),
                    sched.get_last_lr()[0],
                )

        train_loss = run_total / max(1, n_batch)
        metrics = {}
        if val_loader is not None and (epoch + 1) % t.get("val_every", 1) == 0:
            metrics = validate(model, val_loader, cfg, log, distributed=distributed)
            if is_main:
                log.info(
                    "epoch %d val: iou=%.4f f1=%.4f skel_f1=%.4f apls=%.4f breaks=%d",
                    epoch,
                    metrics.get("iou", 0),
                    metrics.get("f1", 0),
                    metrics.get("skel_f1", 0),
                    metrics.get("apls", 0),
                    int(metrics.get("breaks", 0)),
                )

        score = metrics.get("iou", -train_loss)
        better = score > best
        if better:
            best = score

        if is_main:
            record = {"epoch": epoch, "train_loss": train_loss, "sec": time.time() - t0, **metrics}
            history.append(record)
            write_json(out_dir / "history.json", history)

            ckpt = {
                "model": _unwrap(model).state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "cfg": cfg,
            }
            torch.save(ckpt, out_dir / "last.pt")
            if better:
                torch.save(ckpt, out_dir / "best.pt")
                log.info("  -> 保存 best.pt (score=%.4f)", best)

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if is_main:
        log.info("训练完成，best score = %.4f，权重在 %s", best, (out_dir / "best.pt").resolve())


if __name__ == "__main__":
    main()
