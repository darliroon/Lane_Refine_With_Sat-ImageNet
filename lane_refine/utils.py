"""通用工具：随机种子、设备、日志。"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "cuda"):
    """返回 torch.device。返回值不做类型注解，以免在模块层引用 torch。"""
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("lane_refine")


def read_manifest(path: str | os.PathLike) -> list[dict]:
    """读取 JSONL 或 CSV 清单。每行/条至少含 name 与三个路径字段。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"清单不存在: {p}")

    if p.suffix.lower() == ".jsonl":
        items = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    if p.suffix.lower() == ".csv":
        import csv

        with open(p, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"不支持的清单格式 {p.suffix}，请用 .jsonl 或 .csv")


def resolve_paths(items: list[dict], base: str | os.PathLike | None = None) -> list[dict]:
    """把清单里的相对路径按 base 解析为绝对路径。"""
    base = Path(base) if base else None
    out = []
    for it in items:
        d = dict(it)
        for k in ("rgb", "a", "gt"):
            if d.get(k) in (None, "", "null"):
                d[k] = None
            else:
                q = Path(d[k])
                if not q.is_absolute() and base is not None:
                    q = base / q
                d[k] = str(q)
        out.append(d)
    return out


def write_json(path: str | os.PathLike, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
