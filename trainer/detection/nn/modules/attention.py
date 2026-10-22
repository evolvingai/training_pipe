"""Area attention: near-global context at conv-like cost.

The feature map is partitioned into a small number of areas along the vertical
axis; scaled-dot-product attention runs within each area. Positional
information comes from a 7x7 depthwise-separable "position perceiver" conv on
the values (no positional encoding). Attention path uses conv + BN, MLP ratio
is ~1.2. Uses torch SDPA (FlashAttention kernels when available).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.detection.nn.modules.conv import Conv
from trainer.detection.nn.registry import register_module


class AreaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, areas: int = 4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.areas = areas
        self.qkv = Conv(dim, dim * 3, 1, act=False)
        self.pe = Conv(dim, dim, 7, g=dim, act=False)  # 7x7 depthwise position perceiver
        self.proj = Conv(dim, dim, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        v_spatial = qkv[:, 2 * c:]
        pe = self.pe(v_spatial)

        areas = min(self.areas, h)
        if h % areas == 0:
            area_h = h // areas
            # NCHW stores channels before spatial positions. Partitioning areas
            # therefore needs an explicit permutation; a direct reshape mixes
            # Q/K/V channels between areas.
            qkv_areas = (
                qkv.reshape(b, 3, self.num_heads, self.head_dim, areas, area_h, w)
                .permute(0, 4, 1, 2, 5, 6, 3)
                .reshape(b * areas, 3, self.num_heads, area_h * w, self.head_dim)
            )
            q, k, v = qkv_areas.unbind(1)
            out = F.scaled_dot_product_attention(q, k, v)
            out = (
                out.reshape(b, areas, self.num_heads, area_h, w, self.head_dim)
                .permute(0, 2, 5, 1, 3, 4)
                .reshape(b, c, h, w)
            )
        else:
            # Multi-scale training can produce heights that are not divisible
            # by the requested area count. Preserve area attention with nearly
            # equal vertical chunks instead of silently switching to global
            # attention (and changing both semantics and memory complexity).
            outputs = []
            for chunk in qkv.tensor_split(areas, dim=2):
                area_h = chunk.shape[2]
                qkv_area = (
                    chunk.reshape(b, 3, self.num_heads, self.head_dim, area_h, w)
                    .permute(0, 1, 2, 4, 5, 3)
                    .reshape(b, 3, self.num_heads, area_h * w, self.head_dim)
                )
                q, k, v = qkv_area.unbind(1)
                area_out = F.scaled_dot_product_attention(q, k, v)
                outputs.append(
                    area_out.reshape(b, self.num_heads, area_h, w, self.head_dim)
                    .permute(0, 1, 4, 2, 3)
                    .reshape(b, c, area_h, w)
                )
            out = torch.cat(outputs, dim=2)
        return self.proj(out + pe)


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 1.2):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = Conv(dim, hidden, 1)
        self.fc2 = Conv(hidden, dim, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class A2Layer(nn.Module):
    """Attention + MLP with scaled residuals."""

    def __init__(self, dim: int, num_heads: int = 8, areas: int = 4, mlp_ratio: float = 1.2,
                 residual_scale: float = 0.01):
        super().__init__()
        self.attn = AreaAttention(dim, num_heads=num_heads, areas=areas)
        self.mlp = MLP(dim, ratio=mlp_ratio)
        self.gamma1 = nn.Parameter(torch.full((1,), residual_scale))
        self.gamma2 = nn.Parameter(torch.full((1,), residual_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma1 * self.attn(x)
        return x + self.gamma2 * self.mlp(x)


@register_module
class A2Block(nn.Module):
    """R-ELAN-style block whose inner units are area-attention layers."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, areas: int = 4):
        super().__init__()
        c_ = int(c2 * e)
        num_heads = max(1, c_ // 32)
        while c_ % num_heads:
            num_heads -= 1
        self.cv_in = Conv(c1, 2 * c_, 1, 1)
        self.blocks = nn.ModuleList(A2Layer(c_, num_heads=num_heads, areas=areas) for _ in range(n))
        self.cv_out = Conv((2 + n) * c_, c2, 1, 1)
        self.residual = c1 == c2
        self.gamma = nn.Parameter(torch.full((1,), 0.01)) if self.residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv_in(x).chunk(2, 1))
        for block in self.blocks:
            y.append(block(y[-1]))
        out = self.cv_out(torch.cat(y, 1))
        if self.residual:
            out = out + self.gamma * x
        return out
