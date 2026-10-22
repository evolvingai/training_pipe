"""Basic convolution blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.detection.nn.registry import register_module


def autopad(k: int, d: int = 1) -> int:
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2


@register_module
class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU with Conv+BN fusion for inference."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, d: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self) -> None:
        """Fold BN into the conv weights; replaces forward."""
        conv, bn = self.conv, self.bn
        w = conv.weight.clone()
        std = (bn.running_var + bn.eps).sqrt()
        gamma = bn.weight / std
        fused_w = w * gamma.view(-1, 1, 1, 1)
        fused_b = bn.bias - bn.running_mean * gamma
        fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                          conv.padding, groups=conv.groups, dilation=conv.dilation, bias=True)
        fused.weight.copy_(fused_w)
        fused.bias.copy_(fused_b)
        self.conv = fused.to(w.device)
        del self.bn
        self.forward = self.forward_fused  # type: ignore[method-assign]


@register_module
class DWConv(Conv):
    """Depthwise convolution."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, act: bool = True):
        super().__init__(c1, c2, k, s, g=min(c1, c2), act=act)


@register_module
class Concat(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, self.dim)


@register_module
class Upsample(nn.Upsample):
    def __init__(self, scale_factor: float = 2.0, mode: str = "nearest"):
        super().__init__(scale_factor=scale_factor, mode=mode)
