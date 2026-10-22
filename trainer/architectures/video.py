"""Video/action-recognition architectures registered with Trainer."""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.registry import register_architecture


class Conv2Plus1D(nn.Sequential):
    """Factor a 3-D convolution into spatial then temporal convolutions."""

    def __init__(self, input_channels: int, output_channels: int, stride: tuple[int, int, int] = (1, 1, 1)):
        temporal, height, width = stride
        middle = max(input_channels, output_channels)
        super().__init__(
            nn.Conv3d(input_channels, middle, (1, 3, 3), stride=(1, height, width), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(middle), nn.ReLU(inplace=True),
            nn.Conv3d(middle, output_channels, (3, 1, 1), stride=(temporal, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(output_channels), nn.ReLU(inplace=True),
        )


class R2Plus1DBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.body = nn.Sequential(Conv2Plus1D(input_channels, output_channels, stride),
                                  Conv2Plus1D(output_channels, output_channels))
        self.skip = nn.Identity() if input_channels == output_channels and stride == (1, 1, 1) else nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 1, stride=stride, bias=False), nn.BatchNorm3d(output_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(value) + self.skip(value))


class R2Plus1DClassifier(nn.Module):
    def __init__(self, num_classes: int, width: int = 16, dropout: float = 0.2, **_: object):
        super().__init__()
        self.stem = Conv2Plus1D(3, width, stride=(1, 2, 2))
        self.stages = nn.Sequential(
            R2Plus1DBlock(width, width),
            R2Plus1DBlock(width, width * 2, (2, 2, 2)),
            R2Plus1DBlock(width * 2, width * 4, (2, 2, 2)),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(width * 4, num_classes))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # Dataset convention is B,T,C,H,W; Conv3d expects B,C,T,H,W.
        value = clips.permute(0, 2, 1, 3, 4)
        return self.head(self.stages(self.stem(value)))


@register_architecture("video-classification", "r2plus1d-tiny", aliases=("r2plus1d", "action-r2plus1d"),
                       description="CPU-scalable R(2+1)D action-recognition network.",
                       family="R(2+1)D",
                       input_contract="Sampled RGB clip tensor B×T×3×H×W from class folders.",
                       output_contract="Per-action logits and clip-level predictions.",
                       training_strategy="Supervised action recognition with shared spatial transforms and temporal replay.",
                       stages=("video", "clip sampling", "spatial convolution", "temporal convolution", "action logits"))
def build_r2plus1d(num_classes: int, width: int = 16, dropout: float = 0.2, **kwargs: object) -> nn.Module:
    return R2Plus1DClassifier(num_classes, width=width, dropout=dropout, **kwargs)
