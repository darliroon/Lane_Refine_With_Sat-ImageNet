# 后续演进：骨干与结构实验计划

> **读者**：在另一台机器上跑实验的 Code Agent。
> **前提**：[README.md](README.md) 里的部署验收（`scripts/smoke_test.py`）已通过，当前 `resnet34` 配置在真实数据上效果不错，现在要进一步提升。
> **目标**：在不破坏现有流程的前提下，按固定顺序比较更大的预训练骨干与几项低成本改动，选出最优配置。
> **范围**：只改模型侧。输出仍是概率图和掩膜，后处理由使用方自己负责，本文档不涉及。

---

## 0. 先读这几条

1. **这份文档里所有"预期""可能"都是推断，没有在真实数据上验证过。** 实验结果与推断不符时，以实验结果为准，并在 §8 的表里如实记录。
2. **每个配置只跑一轮，不要多种子重复，也不要同时开很多实验臂。** 按 §3 的顺序一次一个，上一步出结果再决定下一步。
3. **每次只改一个因素。** 骨干、通道、损失、分辨率不要在同一次实验里一起变，否则看不出是谁起的作用。
4. **看 `apls`、`breaks`、`skel_f1`，不要看 `iou`。** 细线任务里 IoU 天然很低，详见 README 坑 1b。
5. **遇到下面任何一种情况就停下来汇报，不要自行绕过：** 显存不足且降到 `batch_size=1` 仍不行；代码需要改动的范围超出本文档写明的部分；实验结果与基线相比出现明显异常（例如全部指标为 0）。

---

## 1. 现状与基线

### 1.1 当前代码里的事实

| 项目 | 现状 | 位置 |
|---|---|---|
| 支持的骨干 | 仅 torchvision 的 `resnet18/34/50` | [model.py:49](lane_refine/model.py#L49) |
| 默认配置 | `resnet34`，`decoder_ch=128`，`out_stride=2`，9 通道输入 | [configs/base.yaml](configs/base.yaml) |
| `best.pt` 的选取依据 | 验证集 `iou`，**不是** `apls` | [train.py:304](lane_refine/train.py#L304) |
| 验证集 GT | 不膨胀；`gt_dilate` 只作用于训练集 | [train.py:70-78](lane_refine/train.py#L70-L78) |
| 评估指标 | `iou/f1/skel_f1/breaks/apls`，写入 `history.json` 与 `metrics.json` | [metrics.py](lane_refine/metrics.py) |

**关于 `best.pt` 的选取依据**：它按 `iou` 选，而本文档要求看 `apls`/`breaks`。两者可能不一致，所以比较不同配置时，**不要直接用各自的 `best.pt`**，而是从 `history.json` 里按 §2.2 的规则选 epoch。`last.pt` 是最后一个 epoch 的权重，`best.pt` 是 `iou` 最高的那个 epoch，没有保存其他 epoch。如果 `apls` 最高的 epoch 不是 `iou` 最高的那个，需要改一行代码让 `best.pt` 按 `apls` 选，见 §6.1。

### 1.2 第一步必须做：记录基线

在做任何改动之前，先用现有的 `resnet34` 配置在**真实数据**上训练一轮，把数字记到 §8 的结果表第 0 行。后面所有实验都跟这一行比。

```bash
python -m lane_refine.train --config configs/base.yaml --out runs/exp0_r34 \
  --manifest data/train.jsonl --val_manifest data/val.jsonl
```

如果已经有一轮真实数据上的 `resnet34` 训练结果，并且配置没改过，可以直接用它的 `history.json`，不用重训。但要确认它是用修复后的代码跑的（验证集不膨胀 GT、学习率调度无负值）。旧代码跑出的验证指标偏高，不能作为基线。

---

## 2. 实验规范

### 2.1 每次实验必须相同的部分

- 同一份 `train.jsonl` 和 `val.jsonl`，不要重新划分
- 同一个 `train.seed`（默认 42）
- 同一个 `train.epochs`（默认 40）、`lr` 以外的优化器设置
- 同一个推理阈值 `eval.threshold`（先固定 0.5，§7 单独做阈值扫描）

### 2.2 怎么从一次训练里取结果

训练结束后读 `runs/<exp>/history.json`，每个 epoch 一条记录，含 `iou/f1/skel_f1/apls/breaks`。

取结果的规则（所有实验统一）：

1. 只看最后 10 个 epoch 的记录，不取单个最优 epoch。理由：验证集较小时单 epoch 指标有波动，取最大值会奖励运气。
2. 报告这 10 个 epoch 的 `apls`、`skel_f1`、`breaks` 的**均值**，以及 `apls` 的最大值与最小值。
3. 两个配置的 `apls` 均值相差不到 0.01，视为平手。这是经验阈值，不是统计检验。验证集越小，这个阈值应当越宽。

下面这段脚本可以直接用（只依赖标准库）：

```python
import json, sys, statistics as st
h = json.load(open(sys.argv[1]))
h = [r for r in h if "apls" in r][-10:]
for k in ("apls", "skel_f1", "breaks", "iou"):
    v = [r[k] for r in h]
    print(f"{k:8s} mean={st.mean(v):.4f} min={min(v):.4f} max={max(v):.4f}")
print("epochs:", h[0]["epoch"], "->", h[-1]["epoch"], "| 平均每 epoch 秒数:", round(st.mean(r["sec"] for r in h), 1))
```

注意：`val_every` 默认为 1，每个 epoch 都验证。如果改成大于 1，"最后 10 条记录"覆盖的 epoch 范围会变大，需要在记录里写明。

### 2.3 同时记录资源占用

每次实验记下：峰值显存（`nvidia-smi` 观察，或训练时 `torch.cuda.max_memory_allocated`）、每 epoch 耗时（`history.json` 里的 `sec`）、`batch_size`。这些决定后面能不能扩大。

---

## 3. 实验顺序

按下表逐步做。**每步结束后停下来，把结果填进 §8，再决定是否继续。**

| 步骤 | 实验 | 要改代码吗 | 改动 | 判断 |
|---|---|---|---|---|
| 0 | `resnet34` 基线 | 否 | 无 | 记录基线 |
| 1 | `resnet50` | 否 | `model.encoder: resnet50` | 容量是不是瓶颈 |
| 2a | ConvNeXt-S | 是，§5 | 新增 timm 编码器 | 步骤 1 有明显提升时走这条 |
| 2b | HRNet-W32 | 是，§5 | 新增 timm 编码器 | 步骤 1 没有提升时走这条 |
| 3 | 步骤 2 里较好者的加大版，或 SegFormer-B3 | 是 | 见 §5 | 2 仍在涨时再加大 |

低成本改动（§6）和骨干实验**独立**，可以在任何一步之间插入，但同样一次只改一项。

### 3.1 步骤 1：resnet50

```bash
python -m lane_refine.train --config configs/base.yaml --out runs/exp1_r50 \
  --manifest data/train.jsonl --val_manifest data/val.jsonl
```

配置里把 `model.encoder` 改成 `resnet50`（建议复制一份 `configs/r50.yaml`，不要直接改 `base.yaml`，这样基线配置始终可复现）。

resnet50 的编码器通道数是 256/512/1024/2048，比 resnet34 的 64/128/256/512 宽，解码器第一级的输入变大。**显存会比 resnet34 高，我没有实测过。** 如果 OOM，按 README 坑 4 的顺序降：`batch_size` 减半，`epochs` 不变。`decoder_ch` 先不要动，否则变成两个因素同时改变。

### 3.2 步骤 1 之后怎么分支

| 结果 | 含义 | 下一步 |
|---|---|---|
| resnet50 的 `apls` 或 `breaks` 明显好于基线 | 容量可能是瓶颈 | 走 2a：ConvNeXt-S |
| 差不多，或只好一点点 | 瓶颈大概率不在容量 | 走 2b：HRNet-W32；同时做 §6 的低成本改动 |
| 更差 | 可能是训练设置不适合更深网络，或数据量不足 | 先停下汇报，附上两条训练曲线，不要直接往更大的模型试 |

"明显好于"的判断用 §2.2 的 0.01 阈值。

---

## 4. 候选模型清单

下表的参数量是我凭记忆写的约数，**没有在本机实测**。实际接入后用 `sum(p.numel() for p in m.parameters())` 核对，再记入 §8。

### 4.1 本文档计划试的

| 模型 | 编码器参数量（约） | 预训练 | 接入方式 | 选它的理由 |
|---|---|---|---|---|
| resnet50 | 24M | ImageNet | 现有代码 | 零改动，先看容量是不是瓶颈 |
| ConvNeXt-S | 50M | ImageNet | timm | 现代 CNN 中常用的强基线 |
| HRNet-W32 | 41M | ImageNet | timm | 全程保持高分辨率分支，遥感道路提取里常见。**我没有在你的数据上验证它对细线更好**，这是基于结构的推测 |
| ConvNeXt-B | 89M | ImageNet | timm | 2a 仍在涨时的加大版 |
| SegFormer MiT-B3 | 45M | ImageNet / ADE20K | HuggingFace | 第二档备选 |

### 4.2 暂不试的，以及原因

| 模型 | 暂不试的原因 |
|---|---|
| SAM / SAM2 编码器、DINOv2、DINOv3 | 原生 stride 14 或 16，1–2 px 的线在特征图上基本消失。要用必须再加一条高分辨率卷积旁路，改动大，收益不确定。另外它们只吃 RGB，A 与派生通道要单独的分支注入 |
| 遥感预训练的 ViT（Prithvi、SatMAE 等） | 多为多光谱输入，RGB 用法与授权要逐个核实；同样有 stride 问题 |
| Swin-T/S/B | 需要配 UperNet 类解码器，改动比 SegFormer 大 |
| resnet101/152、wide_resnet、resnext | 与 resnet50 同族，若 resnet50 没有提升，它们大概率也没有 |

关于 ConvNeXt 的提醒：它的第一层是 4×4、stride 4 的不重叠卷积，ResNet 是 7×7、stride 2 的重叠卷积。我推测不重叠的 stem 对 1–2 px 的线可能更吃亏，**但没有验证**。所以它排在 resnet50 之后，且要和 HRNet 对照。

---

## 5. 接入 timm 编码器（步骤 2 之前要做）

### 5.1 目标

在 [model.py](lane_refine/model.py) 里加一个基于 timm 的通用编码器，**保留现有 `ResNetUNet` 路径不变**，通过配置选择：

```yaml
model:
  encoder: timm:convnext_small     # 以 "timm:" 开头走新路径；resnet18/34/50 仍走旧路径
  channels: [rgb, a, skel, endpoint, dir, dist]
  pretrained: true
  out_stride: 2
  decoder_ch: 128
```

### 5.2 要点

1. 用 `timm.create_model(name, pretrained=..., features_only=True, in_chans=9)`。timm 会自动把预训练的第一层卷积适配到 9 通道，但**适配方式（复制并缩放权重）与现有 ResNet 路径的做法（RGB 保留、新增通道取均值）不同**。对比骨干时这个差异本身会影响结果。建议手动处理第一层，与现有路径保持一致：先 `in_chans=3` 加载，再自己把第一层卷积替换成 9 通道。
2. 不同骨干 `feature_info` 给出的各级通道数和 stride 不一样，解码器不能写死。用 `model.feature_info.channels()` 和 `.reduction()` 动态构建 `DecoderBlock`。
3. **ConvNeXt 的特征 stride 是 4/8/16/32，没有 stride 2 的特征；HRNet 的第一个特征是 stride 2。** 现有解码器的最后一级（`dec1`）不使用 skip，所以两者都能接，但要确认输出尺寸与现有路径一致（`H/out_stride`）。
4. 输入尺寸要能被 32 整除。训练瓦片是 256 或 512，没问题；大图推理走 `--tile` 已经满足。
5. `requirements.txt` 加 `timm`。离线机器要提前下载权重，缓存目录在 `~/.cache/huggingface/hub/`（timm 新版本从 HuggingFace Hub 下载）。**我没有核实远端的网络与 timm 版本，先确认能下载。**

### 5.3 接入后必须做的检查

不要直接开训，先过下面几项：

1. 用 `pretrained=False` 跑一次前向，输入 `(2, 9, 256, 256)`，输出应为 `(2, 1, 128, 128)`（`out_stride=2`）。
2. 参数量与 §4 的约数大致相符；差很多说明加载错了模型。
3. 跑 `scripts/smoke_test.py`，旧的 resnet 路径必须仍然通过。
4. 用 `pretrained=True` 加载，确认第一层卷积的 RGB 三个通道权重与官方权重一致（说明预训练确实加载了）。

任何一项不过就停下汇报。

---

## 6. 低成本改动（与骨干实验独立）

这几项改动小、成本低，和换骨干对照着试。**一次只试一项。**

### 6.1 让 `best.pt` 按 `apls` 选

现状按 `iou` 选（[train.py:304](lane_refine/train.py#L304)）。若你更看重拓扑，可以把这一行的 `iou` 换成 `apls`。注意 `apls` 在退化样本上可能是 `nan`，聚合时已用 `nanmean`，一般不会是 `nan`，但 `nan > best` 恒为假，保险起见加一个判断。**这是行为变更，改之前先在 §8 记录原因。** 如果 §2.2 的取结果规则已经够用，可以不改。

### 6.2 最后一级解码器没有 skip

[model.py:126-127](lane_refine/model.py#L126-L127) 里，`x0`（stride 2 的浅层特征，由第一层卷积产生）算出来后没有传给 `dec1`，`dec1` 的 skip 是 `None`。结果是输出虽然是 stride 2，但最后一次上采样没有高分辨率特征补细节。

这可能对 1–2 px 的线有影响，**但我没有验证过。** 改法：把 `x0` 作为 skip 传给 `dec1`，并相应调整 `DecoderBlock` 的 `skip_ch`（resnet 系列 `x0` 的通道数是 64）。这是结构改动，要重新训练，不能加载旧权重。

做这个实验时，基线是步骤 0 的 `resnet34`，只改这一处。

### 6.3 训练瓦片 256 → 512

线宽信息在低分辨率下更容易丢。如果真实数据的原始瓦片大于 256，可以直接用 512 训练。注意这会改变数据，所有骨干实验的基线都要用同一分辨率重新记录，不要混着比。显存和耗时约 4 倍，`batch_size` 要降。

### 6.4 `w_cldice`

默认 1.0。调高会更强调连通性，但可能让输出变粗或出现多余连接。可以试 2.0。同样一次只改这一项，看 `breaks` 与 `skel_f1` 是否一起改善，只有 `breaks` 降而 `skel_f1` 也降，说明是在用精度换连通。

### 6.5 数据量不够时

如果真实数据只有几百张，大骨干的优势会被数据量吃掉。这时 §3 的骨干实验收益可能很小，优先做 6.3、6.4。**数据量到底多大算"不够"，我没有依据给出界限**，以实验为准：resnet50 没提升就是信号。

---

## 7. 阈值扫描（每个最终候选都做）

选定最终候选后，用验证集扫一遍二值化阈值，再定最终阈值：

```bash
for t in 0.3 0.4 0.5 0.6; do
  python -m lane_refine.infer --ckpt runs/<exp>/last.pt \
    --manifest data/val.jsonl --out runs/<exp>/eval_t$t --eval --threshold $t
done
```

读各目录下的 `metrics.json` 的 `summary`。合成数据上的经验是阈值 0.5 附近 `skel_f1` 最高，`0.3/0.4` 明显更差，但真实数据不一定如此，以实测为准。使用方如果直接用概率图（`--save_prob`），阈值只影响本工程的掩膜输出。

---

## 8. 结果记录表

每完成一项，在这张表里加一行。数字按 §2.2 的规则取，不要填单个最优 epoch。

| # | 实验 | 配置文件 | 参数量(M) | batch/卡 | 峰值显存 | s/epoch | apls | skel_f1 | breaks | iou | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | resnet34 基线 | base.yaml | | | | | | | | | |
| 1 | resnet50 | r50.yaml | | | | | | | | | |
| 2a | ConvNeXt-S | | | | | | | | | | |
| 2b | HRNet-W32 | | | | | | | | | | |
| 6.2 | resnet34 + dec1 加 skip | | | | | | | | | | |

"结论"一列只写事实，例如"apls 比基线高 0.03，breaks 从 2.1 降到 1.4，显存增加 6GB"，不要写"效果显著"这类判断。

---

## 9. 汇报时要给的内容

每完成一个步骤，向使用方汇报：

1. §8 表里新增的那一行
2. 该步骤的 `history.json` 路径与训练曲线（`apls`、`breaks` 随 epoch 的变化）
3. 下一步按 §3.2 的分支应该做什么，以及为什么
4. 任何偏离本文档的地方，包括改动了哪些文件，原因是什么

遇到 §0 第 5 条列出的情况，直接停下，不要等做完整个计划再汇报。

---

## 10. 文件改动清单（预期）

| 文件 | 改动 | 何时 |
|---|---|---|
| `configs/r50.yaml` | 新增，复制 `base.yaml` 并改 `model.encoder` | 步骤 1 |
| `lane_refine/model.py` | 新增 timm 编码器路径，保留 `ResNetUNet` | 步骤 2 之前（§5） |
| `requirements.txt` | 加 `timm` | 同上 |
| `configs/convnext_s.yaml`、`configs/hrnet_w32.yaml` | 新增 | 步骤 2 |
| `lane_refine/train.py` | 仅在决定采用 §6.1 时改一行 | 可选 |

除此之外不要改其他文件。需要改的话，先停下汇报。
