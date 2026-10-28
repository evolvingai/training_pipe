"""Anchor-free detection head with decoupled branches and DFL box regression."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from trainer.detection.nn.modules.conv import Conv, DWConv
from trainer.detection.nn.registry import register_module
from trainer.detection.utils.boxes import dist2bbox, make_anchors


class DFL(nn.Module):
    """Distribution Focal Loss decoder: expectation over reg_max bins (frozen conv)."""

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # (B, 4*c1, A)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


@register_module
class Detect(nn.Module):
    """Per-scale decoupled cls/reg convs. Training: raw per-scale maps.
    Eval: additionally decodes to (B, 4 + nc, A) with xywh in input pixels."""

    def __init__(self, nc: int, ch: tuple[int, ...] = (), reg_max: int = 16, reg_channels: int = 0):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.in_channels = tuple(ch)
        self.reg_channels_override = reg_channels
        self.register_buffer("stride", torch.zeros(self.nl), persistent=False)  # set by DetectionModel

        if reg_channels < 0:
            raise ValueError("reg_channels must be nonnegative")
        c2 = reg_channels or max(16, ch[0] // 4, reg_max * 4)
        self.reg_channels = c2
        c3 = max(ch[0], min(nc, 100))
        self.reg_convs = nn.ModuleList(
            nn.Sequential(Conv(c, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for c in ch
        )
        self.cls_convs = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(c, c, 3), Conv(c, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            ) for c in ch
        )
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    def bias_init(self) -> None:
        """Initialize biases: boxes near 1.0, class logits for low prior probability."""
        for reg, cls, s in zip(self.reg_convs, self.cls_convs, self.stride):
            reg[-1].bias.data[:] = 1.0
            cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / float(s)) ** 2)

    def forward(self, xs: list[torch.Tensor]):
        feats = self.forward_raw(xs)
        if self.training:
            return feats
        return self.decode(feats), feats

    def forward_raw(self, xs: list[torch.Tensor]):
        return [torch.cat((self.reg_convs[i](x), self.cls_convs[i](x)), 1) for i, x in enumerate(xs)]

    def decode(self, feats: list[torch.Tensor]) -> torch.Tensor:
        anchors, strides = make_anchors(feats, self.stride, 0.5)
        b = feats[0].shape[0]
        cat = torch.cat([f.view(b, self.no, -1) for f in feats], 2)
        box, cls = cat.split((self.reg_max * 4, self.nc), 1)
        dist = self.dfl(box)
        dbox = dist2bbox(dist, anchors.transpose(0, 1).unsqueeze(0), xywh=True, dim=1) * strides.transpose(0, 1)
        return torch.cat((dbox, cls.sigmoid()), 1)
