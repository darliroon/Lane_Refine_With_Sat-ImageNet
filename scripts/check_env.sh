#!/usr/bin/env bash
# 环境体检。只读。
set -u

echo "=========== 系统 ==========="
uname -srm
[ -f /etc/os-release ] && . /etc/os-release && echo "$PRETTY_NAME"
echo

echo "=========== GPU ==========="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
else
    echo "[WARN] 无 nvidia-smi，将只能用 CPU 训练（很慢，仅适合接口验证）"
fi
echo

echo "=========== Python ==========="
which python; python -V
echo

echo "=========== 依赖 ==========="
python - <<'PY'
import importlib
def chk(m):
    try:
        mod = importlib.import_module(m)
        print(f"[OK]      {m} {getattr(mod,'__version__','?')}")
        return mod
    except Exception as e:
        print(f"[MISSING] {m}: {type(e).__name__}")
        return None

torch = chk("torch"); chk("torchvision"); chk("numpy"); chk("scipy")
chk("skimage"); chk("PIL"); chk("yaml"); chk("networkx")

if torch is not None:
    print()
    print("cuda available:", torch.cuda.is_available())
    print("device count  :", torch.cuda.device_count())
    print("torch.cuda    :", torch.version.cuda)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu{i}: {p.name}  {p.total_memory/1024**3:.1f} GiB")
        print()
        n = torch.cuda.get_device_count()
        total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n))
        if total < 12 * 1024**3:
            print("=> 显存偏小：把 batch_size 降到 2~4，或改用 resnet18 + decoder_ch 64")
        else:
            print("=> 显存充足，可以按配置默认值训练")
PY
echo

echo "=========== 项目自检 ==========="
python - <<'PY'
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent if "__file__" in dir() else pathlib.Path(".")
sys.path.insert(0, str(pathlib.Path.cwd()))
try:
    import lane_refine
    from lane_refine.hint import num_channels
    from lane_refine.model import ResNetUNet
    print("[OK] lane_refine 可导入, 版本", lane_refine.__version__)
    print("[OK] 默认通道数 =", num_channels(["rgb","a","skel","endpoint","dir","dist"]))
except Exception as e:
    print("[FAIL] 导入失败:", e)
    sys.exit(1)

import numpy as np
from lane_refine.hint import compute_hints
from lane_refine.metrics import evaluate
from lane_refine.tiling import make_tiles, fuse

# 1) 条件通道派生
gt = np.zeros((96,96), np.float32); gt[45:47, 10:86] = 1
h = compute_hints(gt)
assert h["skeleton"].sum() > 0 and h["dir"].shape[0] == 2, "hint 派生异常"
print("[OK] hint 派生通道正常（骨架/端点/方向场/距离场）")

# 2) 评估指标
a = gt.copy(); a[43:49, 40:50] = 0
e_gt = evaluate(gt > 0.5, gt > 0.5, compute_apls=True)
e_a = evaluate(a > 0.5, gt > 0.5, compute_apls=True)
assert e_gt["apls"] > 0.999, "GT 自比 APLS 应接近 1"
assert e_a["apls"] < 0.5, "断裂样本 APLS 应明显降低"
print(f"[OK] 评估指标正常: APLS {e_a['apls']:.3f} (GT={e_gt['apls']:.3f}), 断段 {e_a['breaks']}")

# 3) 切块融合
boxes = make_tiles(600, 700, 256, 64)
probs = [np.full((y1-y0, x1-x0), 0.7, np.float32) for y0,y1,x0,x1 in boxes]
out = fuse(probs, boxes, (600,700), 64)
assert abs(out.min() - 0.7) < 1e-5 and abs(out.max() - 0.7) < 1e-5, "切块融合边界异常"
print("[OK] 大图切块融合正常（无黑边）")

print("[OK] 全部自检通过")
PY
