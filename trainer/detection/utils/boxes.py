"""Box format conversions and IoU variants. Boxes are float tensors/arrays.

Formats: xyxy (x1, y1, x2, y2), xywh (center x, center y, w, h),
both absolute pixels unless stated otherwise.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    half_w, half_h = x[..., 2] / 2, x[..., 3] / 2
    y[..., 0] = x[..., 0] - half_w
    y[..., 1] = x[..., 1] - half_h
    y[..., 2] = x[..., 0] + half_w
    y[..., 3] = x[..., 1] + half_h
    return y


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def clip_boxes(boxes, shape):
    """Clip xyxy boxes in-place to image shape (h, w)."""
    h, w = shape[:2]
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, w)
        boxes[..., 1].clamp_(0, h)
        boxes[..., 2].clamp_(0, w)
        boxes[..., 3].clamp_(0, h)
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)
    return boxes


def scale_boxes(boxes, from_shape, to_shape, pad=(0.0, 0.0), gain: float | None = None):
    """Map xyxy boxes from a letterboxed image back to the original image."""
    if gain is None:
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
    boxes = boxes.clone() if isinstance(boxes, torch.Tensor) else np.copy(boxes)
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    return clip_boxes(boxes, to_shape)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU. box1: (N, 4), box2: (M, 4), both xyxy. Returns (N, M)."""
    a1, a2 = box1[:, None, :2], box1[:, None, 2:]
    b1, b2 = box2[None, :, :2], box2[None, :, 2:]
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(-1)
    area1 = (box1[:, 2:] - box1[:, :2]).prod(-1)[:, None]
    area2 = (box2[:, 2:] - box2[:, :2]).prod(-1)[None, :]
    return inter / (area1 + area2 - inter + eps)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = False, ciou: bool = False,
             eps: float = 1e-7) -> torch.Tensor:
    """Elementwise IoU (or CIoU) between aligned boxes of the same shape (..., 4)."""
    if xywh:
        box1, box2 = xywh2xyxy(box1), xywh2xyxy(box2)
    b1x1, b1y1, b1x2, b1y2 = box1.chunk(4, -1)
    b2x1, b2y1, b2x2, b2y2 = box2.chunk(4, -1)
    w1, h1 = b1x2 - b1x1, (b1y2 - b1y1).clamp_(eps)
    w2, h2 = b2x2 - b2x1, (b2y2 - b2y1).clamp_(eps)

    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp_(0) * (
        torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if not ciou:
        return iou.squeeze(-1)

    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2x1 + b2x2 - b1x1 - b1x2).pow(2) + (b2y1 + b2y2 - b1y1 - b1y2).pow(2)) / 4
    v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return (iou - (rho2 / c2 + v * alpha)).squeeze(-1)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = False, dim: int = -1) -> torch.Tensor:
    """Decode (left, top, right, bottom) distances from anchor points to boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(bbox: torch.Tensor, anchor_points: torch.Tensor, reg_max: float) -> torch.Tensor:
    """Encode xyxy boxes as (left, top, right, bottom) distances, clamped to reg_max."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


def make_anchors(feats: list[torch.Tensor], strides, grid_cell_offset: float = 0.5):
    """Anchor centers and per-anchor stride tensors from feature maps."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for f, stride in zip(feats, strides):
        h, w = f.shape[-2:]
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(stride), dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)
