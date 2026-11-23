# 车道线精化：由影像 B 与粗提取 A 预测精化结果 C

> **读者**：在另一台机器上执行任务的编码 Agent。
> **目标**：给定影像瓦片 B(256×256) 与粗提取车道线 A，输出拓扑正确的精化车道线 C。
> **方法**：多通道语义分割（ImageNet 预训练骨干）+ 连通性损失。输出概率图/掩膜，后处理由下游负责。**不用扩散模型。**

---

## 0. 为什么不是扩散

上一版 ControlNet + SD1.5 出"一团乱麻"，是结构不兼容，不是调参问题：

| 问题 | 后果 |
|---|---|
| VAE 8 倍下采样 | 256 瓦片里 1–2 px 宽的车道线进入 latent 后变亚像素，信息在编码阶段就丢了 |
| ControlNet 假设条件图是对的 | A 的断点/错误合流被当作真实结构原样搬运，网络没有"这里要改"的监督 |
| 扩散建模一对多分布 | C 是确定性输出，引入的只有方差 |

因此本方案改用**结构化预测**：模型只负责像素级预测并输出概率图，拓扑修复交给后处理（你们已有自己的后处理）。

---

## 1. 执行顺序

| 步骤 | 命令 | 产物 | 耗时 |
|---|---|---|---|
| 1 | `bash scripts/setup.sh` | 虚拟环境 + 自检 | 5–15 min（视网络） |
| 2 | `bash scripts/check_env.sh` | 环境报告 | 10 s |
| 3 | **`python scripts/smoke_test.py`** | 全链路验收 | CPU 1–3 min / GPU 30 s |
| 4 | `python scripts/make_synthetic.py --out data_synth` | 合成数据 | 10 s |
| 5 | `python scripts/prepare_manifest.py ...` | 真实数据清单 | 秒级 |
| 6 | 训练 | `runs/base/best.pt` | 见 §6 |
| 7 | 推理 + 评估 | `runs/eval/metrics.json` | 视数据量 |

**第 3 步是部署验收关卡，必须通过再往下走。** 它自动完成"生成合成数据 → 训练 1 epoch → 推理 → 评估"，任何环节有依赖或代码问题都会在这里暴露，且不需要真实数据、不下载预训练权重。

如果第 3 步失败，把完整输出贴出来，不要直接跳到真实数据训练。

---

## 2. 环境

### 2.1 硬件

| 配置 | 可行性 |
|---|---|
| 1× 24 GB（3090/4090） | 推荐。512 瓦片 bs 8 约 9 GB |
| 1× 12 GB（2080Ti/3060） | 可行，`batch_size=4` |
| 1× 8 GB | 可行，`batch_size=2` + `decoder_ch=64` |
| 纯 CPU | 只能跑通流程，训练不可行 |

### 2.2 安装

```bash
cd lane_refine
bash scripts/setup.sh
```

脚本会建 `.venv`、装 [requirements.txt](requirements.txt)、跑自检。重复执行是安全的。

手动安装等价于：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

torch 首次下载约 2 GB。若机器已有可用的 torch（2.1+），可注释掉 requirements 里的 torch 行，避免重复下载。

### 2.3 离线环境

`configs/base.yaml` 里 `model.pretrained: true` 会下载 torchvision 的 ImageNet 权重（resnet34 约 75 MB），缓存在 `~/.cache/torch/hub/checkpoints/`。

完全断网的机器：

```bash
# 在有网机器上
python -c "import torchvision; torchvision.models.resnet34(weights='IMAGENET1K_V1')"
# 拷贝 ~/.cache/torch/hub/checkpoints/resnet34-*.pth 到目标机同名路径
```

或直接把配置改成 `pretrained: false`（不推荐，小数据下收敛慢很多）。

---

## 3. 数据

### 3.1 目录约定

```
data/
  B/    0001.png ...    影像瓦片
  A/    0001.png ...    粗提取车道线（二值）
  GT/   0001.png ...    精化标注（二值）
```

文件名 stem 是配对键。掩膜读入规则：**灰度值 > 0 即视为前景**，所以 0/255、0/1 都行。

尺寸要求：A、B、GT 三者应同尺寸。不一致时 A/GT 会被最近邻缩放到 B 的尺寸并打警告——**不要依赖这个行为**，提前对齐。

### 3.2 生成清单

```bash
python scripts/prepare_manifest.py \
  --b data/B --a data/A --gt data/GT \
  --out data/train.jsonl --split-train 0.9
```

产出 `data/train.jsonl` 与 `data/val.jsonl`，每行：

```json
{"name": "0001", "rgb": "B/0001.png", "a": "A/0001.png", "gt": "GT/0001.png"}
```

路径相对于 `--root`（默认是 B 的父目录）。整个 `data/` 搬走后不用改清单。

### 3.3 数据量

| 量级 | 预期 |
|---|---|
| < 500 | 能跑，但建议只用 `rgb + a` 四通道，别上重模型 |
| 500–5000 | 本方案的甜点区 |
| 5000–50000 | 按默认配置，考虑加大 `epochs` 和 `decoder_ch` |

### 3.4 数据检查清单（做之前先查一遍）

- [ ] A 和 GT **没有装反**（A 应该是"有断裂、有毛刺"的那张）
- [ ] GT 线宽一致（通常 2–3 px）。线宽不一致会直接污染 `gt_dilate` 的语义
- [ ] A 的断裂属于"结构缺失"而非"整块瓦片没有路"（后者应作为负样本，不是精化任务）
- [ ] 瓦片是否已经对齐（同一地理位置，无偏移）

---

## 4. 训练

```bash
python -m lane_refine.train --config configs/base.yaml --out runs/base
```

多卡：

```bash
NPROC=4 bash scripts/train_ddp.sh
```

常用覆盖：

```bash
python -m lane_refine.train --config configs/base.yaml --out runs/base \
  --manifest data/train.jsonl --val_manifest data/val.jsonl \
  --epochs 60
```

### 4.1 关键配置项

完整项见 [configs/base.yaml](configs/base.yaml)。以下是决定成败的几个：

| 键 | 默认 | 作用 |
|---|---|---|
| `model.channels` | `[rgb, a, skel, endpoint, dir, dist]` | **最重要的设计**。`endpoint` 直接标出"这里要连"，`dir` 提供合流处的朝向消歧，这两个对断裂/合流改善最大 |
| `model.out_stride` | `2` | 输出分辨率 = 输入/2。不要改成 4，1–2 px 的线在 stride 4 上必然断 |
| `data.gt_dilate` | `1` | GT **训练时**膨胀 1 px。细线任务里能显著稳定早期训练。**只作用于训练集**，验证集始终用原始 GT——否则指标会虚高（实测 IoU 0.502 vs 真实 0.265） |
| `data.hint_dropout` | `0.15` | 整张 A 派生通道随机置零。防止网络无脑复制 A 的错误 |
| `train.w_cldice` | `1.0` | **clDice 权重**。它直接优化骨架连通性，是断裂问题的主要解药。设为 0 可做消融 |
| `train.pos_weight` | `5.0` | 车道线像素占比通常 < 1%，需要正样本加权 |

### 4.2 消融顺序（每步都该跑一次验证）

```bash
# A. 只用 rgb+a（对照组，代表"不用派生通道"）
#    改 configs/base.yaml: channels: [rgb, a]

# B. 加 skel+endpoint（测端点通道的贡献）
#    channels: [rgb, a, skel, endpoint]

# C. 全通道（本方案默认）

# D. 去掉 clDice（测连通性损失的贡献）
#    train.w_cldice: 0.0
```

这四组跑完，你就知道每个设计各值多少。**不要只跑 C 就下结论。**

### 4.3 参照基线

在合成数据上实测过（16 样本 / 128px / resnet18 / CPU 25 epoch）：

**A 相对 GT** 的指标，即"什么都不做"的下限：

| 指标 | 合成数据上 A vs GT |
|---|---|
| f1 | 0.888 |
| skel_f1 | 0.901 |
| breaks | 5.9（GT 自身 0.3） |
| **apls** | **0.271** |

**训练后**同配置达到：

| 指标 | 训练后 |
|---|---|
| f1 | 0.419 |
| skel_f1 | 0.945 |
| breaks | 0.3 → **0** |
| **apls** | **0.267 → 0.957** |

判读要点：

- **APLS 是最能说明问题的指标**（0.27 → 0.96），它反映拓扑是否连通、路径是否合理
- **IoU 在细线任务上天然很低**（这个任务 GT 只有约 2 px 宽，模型预测 1–2 px，错开半像素 IoU 就掉一半）。训练后 IoU 约 0.26 是正常的，**不要拿 IoU 0.5+ 当成及格线**
- 判断模型是否学到东西：看 `apls` 和 `breaks`，不要看 `iou`

真实数据跑出来量级应该接近，否则说明数据或读取有问题。

### 4.4 观察什么

日志每 epoch 输出：

```
epoch 3 val: iou=0.8213 f1=0.8901 skel_f1=0.9455 apls=0.7120 breaks=2
```

| 指标 | 含义 | 目标 |
|---|---|---|
| `iou` / `f1` | 像素准确度 | 越高越好，但会奖励画粗线 |
| `skel_f1` | 骨架 F1（容差 2 px） | 比像素 F1 更贴近"线画对了没有" |
| `apls` | 路径拓扑一致性 [0,1] | 1 = 拓扑完全一致。**断裂和错误合流会直接拉低它** |
| `breaks` | 骨架断段数 | 越小越好，`break_excess` = 预测 − GT |

**判读准则**：

- `f1` 高但 `skel_f1` 低 → 线画粗了，把 `gt_dilate` 降到 0，或检查 GT 线宽
- `skel_f1` 高但 `breaks` 大 → 位置对但断，加大 `w_cldice` 或调 `gt_dilate`
- `apls` 明显低于 `skel_f1` → 拓扑错（错误合流/连接）。注意：模型只输出概率，**拓扑最终由你的后处理决定**，这里反映的是"模型给出的结构对最终结果的支持程度"
- `best.pt` 按 `iou` 选。若你更关心拓扑，把 `train.py:score = metrics.get("iou", ...)` 改成 `apls`

产物：

```
runs/base/
  best.pt          # 按验证分数选出的权重（推理用这个）
  last.pt          # 最后一个 epoch
  history.json     # 每个 epoch 的完整指标
  config_used.yaml # 实际生效的配置（含所有覆盖）
```

`best.pt` 里存了 `cfg`，推理时不需要再传配置。

### 4.5 已知局限

这些是设计上的取舍，不是 bug，但会影响指标读法：

| 局限 | 影响 | 说明 |
|---|---|---|
| APLS 用随机采样近似 | 对"多点捷径"不够敏感（实测 U 形加一条抄近路，APLS 仍 0.946） | 它主要抓**断裂**（断开的线实测 0.15），对局部多余连接较弱。要严格度量拓扑请用标准 APLS 实现 |
| 方向场用高斯尺度 sigma=2 | 极细（1 px）且急转弯处方向估计偏软 | 调 `hint.direction_field(sigma=)` |
| 模型不负责拓扑 | 断口过大时模型可能直接接不上 | 拓扑修复在你已有的后处理里；模型的价值是把概率提上去、把断口缩小到后处理能处理的范围 |
| 指标基于骨架 8 连通 | 斜向线有轻微锯齿误差 | 对比较训练不同配置影响不大 |

### 4.6 从 checkpoint 恢复

```bash
python -m lane_refine.train --config configs/base.yaml --out runs/base --resume runs/base/last.pt
```

---

## 5. 推理

### 5.1 单张

```bash
python -m lane_refine.infer --ckpt runs/base/best.pt \
  --rgb data/B/0001.png --a data/A/0001.png \
  --out out/
```

产出 `out/0001_C.png`。

### 5.2 批量 + 评估

```bash
python -m lane_refine.infer --ckpt runs/base/best.pt \
  --manifest data/val.jsonl --out runs/eval --eval --save_prob
```

产出：

```
runs/eval/
  0001_C.png        # 精化结果（二值掩膜）
  0001_prob.npy     # float16 概率图，供你的后处理使用
  metrics.json      # summary + per_tile
```

`metrics.json` 的 `summary` 是和 GT 对比的平均值，直接拿去和 §4.3 的基线对比。

### 5.3 大图（超过训练瓦片尺寸）

```bash
python -m lane_refine.infer --ckpt runs/base/best.pt \
  --rgb big_B.png --a big_A.png --out out_big/ \
  --tile 512 --overlap 128
```

重叠窗口 + 余弦窗加权融合，消除接缝。`overlap` 建议 ≥ 1/4 tile。

### 5.4 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--threshold` | 0.5 | 二值化阈值。**扫一遍 0.3–0.7 找最优**，通常 0.4 左右对细线更友好 |
| `--save_prob` | 关 | 输出 float16 概率图 `.npy`，**推荐开启**——后处理在概率图上做比在二值图上做更细粒度 |
| `--tile` / `--overlap` | 0 / 128 | 大图切块尺寸与重叠 |

### 5.5 与你的后处理对接

本工程**只负责模型部分**，输出为：

- `*_C.png`：二值掩膜（0/255），可直接喂给已有后处理
- `*_prob.npy`：float16 概率图，形状 `(H, W)`，值域 `[0,1]`

如果你的后处理有内部阈值或多阈值决策，**用概率图**：把 `binarize` 那一步搬到后处理里，可以按区域自适应。二值掩膜只是按固定 `--threshold` 切出来的便利产物。

跨瓦片的后处理（矢量层合并）在你们侧完成——本工程的切块推理只保证概率图无接缝。

---

## 6. 耗时参考

| 规模 | 单卡 3090 | 4×3090 |
|---|---|---|
| 2000 对 × 256px，40 epoch | 约 1.5 h | 约 30 min |
| 2000 对 × 512px，40 epoch | 约 4 h | 约 1.2 h |
| 10000 对 × 512px，60 epoch | 约 30 h | 约 8 h |

显存不够时的降级顺序：`batch_size` → `decoder_ch` → `encoder: resnet18`。**不要降 `out_stride`**。

---

## 7. 故障排查

### 坑 1：`f1` 很高但结果线断

`gt_dilate` 太大会让模型学会画粗线，骨架化后仍断。把 `data.gt_dilate` 设为 0，并确认 GT 本身线宽 ≤ 3 px。

### 坑 1b：`iou` 只有 0.2–0.3，是不是模型没用？

**不是**。细线任务里 IoU 天然低：GT 只有 2 px 宽，预测错开半个像素 IoU 就掉一半。我在合成数据上实测同一份预测——`skel_f1` 0.945、`apls` 0.957、`breaks` 0 的同时，`iou` 只有 0.265。

判断标准用 **`apls` 和 `breaks`**，不要用 `iou`。如果 `skel_f1` > 0.9 且 `breaks` 接近 GT，模型就是有效的。

### 坑 2：训练 loss 下降但 `apls` 不涨

正常，而且**不应该靠模型解决**。像素损失管不到拓扑，模型能提供的是更高的 `skel_f1` 和更小的断口——拓扑最终由你的后处理决定。判断模型是否有效，看 `skel_f1` 和 `breaks` 有没有改善，而不是看 `apls`。

### 坑 3：`hint_dropout` 导致训练不稳

0.15 通常安全。数据量 < 300 时降到 0.05；数据量大且发现模型过度依赖 A 时提到 0.25。

### 坑 4：显存 OOM

```
batch_size 8 -> 4 -> 2  (同步调大 epochs 保持总步数)
decoder_ch 128 -> 64
encoder resnet34 -> resnet18
```

不要通过降输入分辨率解决——线宽信息在低分辨率下直接丢失。

### 坑 5：`model.pretrained=true` 但下载失败

见 §2.3。临时应急：配置改 `pretrained: false`，训练能跑但收敛慢。

### 坑 6：验证集指标波动大

验证集 < 50 张时指标噪声很大。用 k-fold 或至少保证验证集 ≥ 100 张。

---

## 8. 项目结构

```
lane_refine/
  configs/base.yaml        训练配置（所有可调项）
  requirements.txt
  lane_refine/
    hint.py                A -> 骨架/端点/方向场/距离场
    imops.py               图像基础操作（无 torch 依赖）
    data.py                数据集与增强
    model.py               骨干 + U-Net 解码器
    losses.py              BCE + Dice + clDice
    metrics.py             IoU / 骨架 F1 / 断段数 / APLS-lite
    tiling.py              大图切块与融合
    train.py               训练入口（含 DDP）
    infer.py               推理入口（输出掩膜 + 概率图）
  scripts/
    setup.sh               一键装环境
    check_env.sh           环境体检 + 自检
    smoke_test.py          全链路冒烟测试（部署验收）
    prepare_manifest.py    目录 -> 清单
    make_synthetic.py      合成数据（自检用）
    train_ddp.sh           多卡启动
```

---

## 9. 后续改进方向（按性价比排序）

1. **加大分辨率**：256 → 512，通常比换大模型收益大
2. **提高连通性损失权重** `w_cldice`：模型层面能做的、对断裂最直接的改进
3. **置信度输出**：`--save_prob` 存概率图，后处理可据此决定哪些区域需要更强修复或人工复核
4. **换骨干**：SegFormer / ConvNeXt。收益通常小于前两项
5. **多尺度输出**：同时监督 stride 2 和 stride 4，缓解细线在小目标上的不稳定
