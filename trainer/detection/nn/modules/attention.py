"""Area attention: near-global context at conv-like cost.

The feature map is partitioned into adaptive vertical or horizontal areas;
scaled-dot-product attention runs within each area. Shifted partitions and
alternating axes let neighboring layers exchange context across area borders. Positional
information comes from a 7x7 depthwise-separable "position perceiver" conv on
the values (no positional encoding). Attention path uses conv + BN, MLP ratio
is ~1.2. Uses torch SDPA (FlashAttention kernels when available).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.detection.nn.modules.conv import Conv
from trainer.detection.nn.modules.relan import C3k
from trainer.detection.nn.registry import register_module


class AreaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, areas: int = 4, axis: str = "vertical",
                 shift: bool = False, max_tokens: int = 512):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        if axis not in ("vertical", "horizontal"):
            raise ValueError("attention axis must be 'vertical' or 'horizontal'")
        if areas <= 0 or max_tokens <= 0:
            raise ValueError("areas and max_tokens must be positive")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.areas = areas
        self.axis = axis
        self.shift = shift
        self.max_tokens = max_tokens
        self.qkv = Conv(dim, dim * 3, 1, act=False)
        self.pe = Conv(dim, dim, 7, g=dim, act=False)  # 7x7 depthwise position perceiver
        self.proj = Conv(dim, dim, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transposed = self.axis == "horizontal"
        if transposed:
            x = x.transpose(2, 3)
        b, c, h, w = x.shape
        areas = min(max(self.areas, math.ceil(h * w / self.max_tokens)), h)
        shift = max(h // areas // 2, 1) if self.shift and h > areas else 0
        if shift:
            x = torch.roll(x, shifts=-shift, dims=2)
        qkv = self.qkv(x)
        v_spatial = qkv[:, 2 * c:]
        pe = self.pe(v_spatial)

        if h % areas == 0:
            area_h = h // areas
            # NCHW stores channels before spatial positions. Explicitly move
            # the area axis ahead of Q/K/V; a direct reshape mixes Q/K/V data.
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
            # Multi-scale inputs are not always divisible by the configured
            # area count. Use nearly equal vertical chunks rather than silently
            # changing to global attention with very different memory cost.
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
        out = self.proj(out + pe)
        if shift:
            out = torch.roll(out, shifts=shift, dims=2)
        return out.transpose(2, 3) if transposed else out


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
                 residual_scale: float = 0.01, axis: str = "vertical", shift: bool = False,
                 max_tokens: int = 512):
        super().__init__()
        self.attn = AreaAttention(
            dim, num_heads=num_heads, areas=areas, axis=axis, shift=shift, max_tokens=max_tokens,
        )
        self.mlp = MLP(dim, ratio=mlp_ratio)
        self.gamma1 = nn.Parameter(torch.full((1,), residual_scale))
        self.gamma2 = nn.Parameter(torch.full((1,), residual_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma1 * self.attn(x)
        return x + self.gamma2 * self.mlp(x)


@register_module
class A2Block(nn.Module):
    """R-ELAN-style block whose inner units are area-attention layers."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, areas: int = 4,
                 max_tokens: int = 512):
        super().__init__()
        c_ = int(c2 * e)
        num_heads = max(1, c_ // 32)
        while c_ % num_heads:
            num_heads -= 1
        self.cv_in = Conv(c1, 2 * c_, 1, 1)
        self.blocks = nn.ModuleList(
            A2Layer(
                c_, num_heads=num_heads, areas=areas,
                axis="vertical" if i % 2 == 0 else "horizontal",
                shift=True, max_tokens=max_tokens,
            )
            for i in range(n)
        )
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


class PaperAreaAttention(nn.Module):
    """Area attention used by the published YOLOv12 architecture.

    Unlike the older adaptive Trainer block above, this keeps the paper graph's
    fused QKV projection, fixed token-area partition and 7x7 depthwise
    position perceiver. PyTorch SDPA selects the best available attention kernel.
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by the head count")
        if area < 1:
            raise ValueError("attention area count must be positive")
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = Conv(dim, dim * 3, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 7, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = height * width
        if tokens % self.area:
            raise ValueError(
                f"feature-map token count {tokens} must be divisible by area={self.area}"
            )
        qkv_map = self.qkv(x)
        qkv = qkv_map.flatten(2).transpose(1, 2)

        area_batch = batch * self.area
        area_tokens = tokens // self.area
        qkv = qkv.reshape(area_batch, area_tokens, channels * 3)
        queries, keys, values = qkv.chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                area_batch, area_tokens, self.num_heads, self.head_dim,
            ).permute(0, 2, 1, 3)

        attended = F.scaled_dot_product_attention(heads(queries), heads(keys), heads(values))
        attended = attended.permute(0, 2, 1, 3).reshape(area_batch, area_tokens, channels)
        attended = attended.reshape(batch, tokens, channels).transpose(1, 2)
        attended = attended.reshape(batch, channels, height, width)
        value_map = values.reshape(batch, tokens, channels).transpose(1, 2)
        value_map = value_map.reshape(batch, channels, height, width)
        return self.proj(attended + self.pe(value_map))


class PaperAttentionBlock(nn.Module):
    """Two-residual attention/MLP unit inside the paper A2C2f block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2,
                 area: int = 1):
        super().__init__()
        self.attn = PaperAreaAttention(dim, num_heads, area)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, hidden, 1), Conv(hidden, dim, 1, act=False))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


@register_module
class A2C2f(nn.Module):
    """Paper-faithful R-ELAN/A2C2f aggregation block."""

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True,
                 area: int = 1, residual: bool = False, mlp_ratio: float = 2.0,
                 e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        hidden = int(c2 * e)
        if hidden % 32:
            raise ValueError("A2C2f hidden channels must be a multiple of 32")
        num_heads = hidden // 32
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv((1 + n) * hidden, c2, 1, 1)
        self.gamma = (
            nn.Parameter(torch.full((c2,), 0.01)) if a2 and residual else None
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                PaperAttentionBlock(hidden, num_heads, mlp_ratio, area),
                PaperAttentionBlock(hidden, num_heads, mlp_ratio, area),
            )
            if a2 else C3k(hidden, hidden, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(block(y[-1]) for block in self.blocks)
        out = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * out
        return out
