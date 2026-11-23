"""分割网络：ImageNet 预训练 ResNet 编码器 + U-Net 解码器，输出 stride 2。

为什么不是 stride 4/8：车道线在 256/512 瓦片里只有 1–2 像素宽，
解码器精度必须到 stride 2，否则亚像素的线在输出栅格上就断了。
推理时再把 logits 双线性上采样回原分辨率。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_act(in_ch: int, out_ch: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride, k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = _conv_bn_act(in_ch + skip_ch, out_ch)
        self.conv2 = _conv_bn_act(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class ResNetUNet(nn.Module):
    """ResNet-18/34/50 编码器 + U-Net 解码器。

    Args:
        encoder: resnet18 | resnet34 | resnet50
        in_channels: 输入通道数（RGB 3 + 条件通道）
        pretrained: 加载 torchvision ImageNet 权重
        out_stride: 2 或 4，输出分辨率 = 输入 / out_stride
        decoder_ch: 解码器基础通道
    """

    SUPPORTED = ("resnet18", "resnet34", "resnet50")

    def __init__(
        self,
        encoder: str = "resnet34",
        in_channels: int = 7,
        pretrained: bool = True,
        out_stride: int = 2,
        decoder_ch: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        if encoder not in self.SUPPORTED:
            raise ValueError(f"encoder 必须是 {self.SUPPORTED}，收到 {encoder}")
        if out_stride not in (2, 4):
            raise ValueError("out_stride 只支持 2 或 4")
        self.out_stride = out_stride

        import torchvision

        weights = None
        if pretrained:
            key = {
                "resnet18": "ResNet18_Weights",
                "resnet34": "ResNet34_Weights",
                "resnet50": "ResNet50_Weights",
            }[encoder]
            weights = getattr(torchvision.models, key).IMAGENET1K_V1

        net = getattr(torchvision.models, encoder)(weights=weights)

        # 替换首层卷积以接纳额外条件通道
        old = net.conv1
        new_conv = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            new_conv.weight.zero_()
            n_copy = min(3, in_channels)
            # 新增通道用 RGB 权重均值初始化：零初始化会让条件通道前期完全失效
            new_conv.weight[:, :n_copy] = old.weight[:, :n_copy]
            if in_channels > 3:
                mean_w = old.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:] = mean_w.expand(-1, in_channels - 3, -1, -1)
        self.conv1 = new_conv
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1  # stride 4
        self.layer2 = net.layer2  # stride 8
        self.layer3 = net.layer3  # stride 16
        self.layer4 = net.layer4  # stride 32

        if encoder in ("resnet18", "resnet34"):
            c1, c2, c3, c4 = 64, 128, 256, 512
        else:
            c1, c2, c3, c4 = 256, 512, 1024, 2048

        d = decoder_ch
        self.dec4 = DecoderBlock(c4, c3, d)
        self.dec3 = DecoderBlock(d, c2, d)
        self.dec2 = DecoderBlock(d, c1, d)
        if out_stride == 2:
            self.dec1 = DecoderBlock(d, 0, d // 2)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(d // 2 if out_stride == 2 else d, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 (N, 1, H/out_stride, W/out_stride) 的 logits。"""
        x0 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x0)
        f1 = self.layer1(x)  # /4
        f2 = self.layer2(f1)  # /8
        f3 = self.layer3(f2)  # /16
        f4 = self.layer4(f3)  # /32

        u = self.dec4(f4, f3)
        u = self.dec3(u, f2)
        u = self.dec2(u, f1)
        if self.out_stride == 2:
            u = self.dec1(u, None)
        u = self.dropout(u)
        return self.head(u)


def build_model(cfg: dict) -> ResNetUNet:
    return ResNetUNet(
        encoder=cfg.get("encoder", "resnet34"),
        in_channels=cfg["in_channels"],
        pretrained=cfg.get("pretrained", True),
        out_stride=cfg.get("out_stride", 2),
        decoder_ch=cfg.get("decoder_ch", 128),
        dropout=cfg.get("dropout", 0.0),
    )
