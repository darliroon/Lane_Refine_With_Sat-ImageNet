"""Lane refinement: 由粗提取车道线 A 与影像 B 预测精化车道线 C。

模块划分
    hint       A -> 派生条件通道（端点、方向场）
    data       数据集与配对清单
    model      分割网络（ResNet 编码器 + U-Net 解码器）
    losses     BCE / Dice / clDice 组合损失
    metrics    像素指标 + 拓扑评估指标（断段数、APLS-lite）
    tiling     大图重叠切块与概率图融合
"""

__version__ = "0.1.0"
