"""Image-classification architectures registered with Trainer."""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.registry import register_architecture


class LayerNorm2d(nn.LayerNorm):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.permute(0, 2, 3, 1)
        value = super().forward(value)
        return value.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, layer_scale: float = 1e-6):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.expand = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.project = nn.Linear(expansion * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value).permute(0, 2, 3, 1)
        value = self.project(self.act(self.expand(self.norm(value))))
        return residual + (self.gamma * value).permute(0, 3, 1, 2)


class ConvNeXtClassifier(nn.Module):
    """Compact ConvNeXt with configurable width/depth for local and VM runs."""

    def __init__(self, num_classes: int, width: int = 32, depths: tuple[int, ...] = (1, 1, 3, 1),
                 dropout: float = 0.1):
        super().__init__()
        dims = [width, width * 2, width * 4, width * 8]
        self.downsample = nn.ModuleList([nn.Sequential(nn.Conv2d(3, dims[0], 4, stride=4), LayerNorm2d(dims[0]))])
        for previous, current in zip(dims[:-1], dims[1:], strict=True):
            self.downsample.append(nn.Sequential(LayerNorm2d(previous), nn.Conv2d(previous, current, 2, stride=2)))
        self.stages = nn.ModuleList([
            nn.Sequential(*(ConvNeXtBlock(dim) for _ in range(depth))) for dim, depth in zip(dims, depths, strict=True)
        ])
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dims[-1], num_classes))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = image
        for downsample, stage in zip(self.downsample, self.stages, strict=True):
            value = stage(downsample(value))
        return self.head(self.norm(value.mean((-2, -1))))


class TinyImageCNN(nn.Module):
    def __init__(self, num_classes: int, width: int = 24, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        previous = 3
        for channels in (width, width * 2, width * 4):
            layers.extend([nn.Conv2d(previous, channels, 3, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(channels), nn.SiLU(inplace=True)])
            previous = channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(previous, num_classes))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image))


@register_architecture("classification", "convnext-tiny", aliases=("convnext",),
                       description="Compact ConvNeXt with depthwise 7x7 blocks and hierarchical downsampling.",
                       family="ConvNeXt",
                       input_contract="RGB image tensor B×3×H×W from class folders.",
                       output_contract="Per-class logits, probabilities, top-k predictions, calibrated thresholds, signed Grad-CAM and stage activations.",
                       training_strategy="Supervised image classification with shared augmentation, EMA and early stopping.",
                       stages=("image", "stem", "hierarchical features", "global pooling", "class logits"))
def build_convnext(num_classes: int, width: int = 32, dropout: float = 0.1, **_: object) -> nn.Module:
    return ConvNeXtClassifier(num_classes, width=width, dropout=dropout)


@register_architecture("classification", "tiny-cnn", description="Minimal convolutional compatibility baseline.",
                       family="CNN baseline",
                       input_contract="RGB image tensor B×3×H×W from class folders.",
                       output_contract="Per-class logits and top-k predictions.",
                       training_strategy="Small supervised baseline for compatibility checks and CPU trials.",
                       stages=("image", "convolution stack", "global pooling", "class logits"))
def build_tiny_cnn(num_classes: int, width: int = 24, dropout: float = 0.1, **_: object) -> nn.Module:
    return TinyImageCNN(num_classes, width=width, dropout=dropout)
