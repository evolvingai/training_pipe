"""Spatial pooling blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.detection.nn.modules.conv import Conv
from trainer.detection.nn.registry import register_module


@register_module
class SPPF(nn.Module):
    """Spatial pyramid pooling (fast): serial max-pools concatenated."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        for _ in range(3):
            y.append(self.pool(y[-1]))
        return self.cv2(torch.cat(y, 1))
