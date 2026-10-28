"""R-ELAN: residual efficient layer aggregation block.

A stack of bottleneck branches whose intermediate outputs are aggregated by
concatenation, plus a scaled residual shortcut from the block input.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.detection.nn.modules.conv import Conv
from trainer.detection.nn.registry import register_module


class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class PaperBottleneck(nn.Module):
    """Configurable bottleneck used by the paper-faithful C3k2 blocks."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 kernels: tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, kernels[0], 1)
        self.cv2 = Conv(hidden, c2, kernels[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """Two-convolution CSP aggregation used as the base of C3k2."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.blocks = nn.ModuleList(
            PaperBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.blocks)
        return self.cv2(torch.cat(y, 1))


class C3k(nn.Module):
    """Three-convolution CSP block with configurable spatial kernels."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True,
                 g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1, 1)
        self.blocks = nn.Sequential(
            *(PaperBottleneck(hidden, hidden, shortcut, g, (k, k), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), 1))


@register_module
class C3k2(C2f):
    """YOLOv12 C3k2 block, including its scale-dependent C3k branch."""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False,
                 e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.blocks = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k else PaperBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


@register_module
class RELAN(nn.Module):
    """Residual ELAN block: split -> n bottlenecks -> aggregate -> scaled residual."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, residual_scale: float = 0.01):
        super().__init__()
        c_ = int(c2 * e)
        self.cv_in = Conv(c1, 2 * c_, 1, 1)
        self.blocks = nn.ModuleList(Bottleneck(c_, c_) for _ in range(n))
        self.cv_out = Conv((2 + n) * c_, c2, 1, 1)
        self.residual = c1 == c2
        self.gamma = nn.Parameter(torch.full((1,), residual_scale)) if self.residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv_in(x).chunk(2, 1))
        for block in self.blocks:
            y.append(block(y[-1]))
        out = self.cv_out(torch.cat(y, 1))
        if self.residual:
            out = out + self.gamma * x
        return out


@register_module
class CSP(nn.Module):
    """Plain CSP block (no attention): alternate lightweight backbone unit."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
