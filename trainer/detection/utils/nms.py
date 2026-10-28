"""In-house non-maximum suppression (no torchvision)."""

from __future__ import annotations

import torch

from trainer.detection.utils.boxes import box_iou, xywh2xyxy


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    """NMS via torchvision (C++/CUDA). boxes (N, 4) xyxy, scores (N,). Returns kept indices."""
    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes.float(), scores.float(), iou_thres)
    except ImportError:
        # fallback: pure-Python greedy NMS
        order = scores.argsort(descending=True)
        keep: list[int] = []
        while order.numel() > 0:
            i = order[0]
            keep.append(int(i))
            if order.numel() == 1:
                break
            ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
            order = order[1:][ious <= iou_thres]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    max_det: int = 300,
    agnostic: bool = False,
    per_class_conf: torch.Tensor | None = None,
    max_nms: int = 30000,
    multi_label: bool = False,
) -> list[torch.Tensor]:
    """Batched NMS on raw head output.

    prediction: (B, 4 + nc, A) with xywh boxes and per-class scores.
    per_class_conf: optional (nc,) tensor of per-class confidence thresholds
        overriding the global conf_thres.
    Returns a list of (n, 6) tensors [x1, y1, x2, y2, conf, cls] per image.
    """
    bs, no, _ = prediction.shape
    nc = no - 4
    outputs: list[torch.Tensor] = []
    for xi in range(bs):
        x = prediction[xi].T  # (A, 4+nc)
        boxes = xywh2xyxy(x[:, :4])
        scores = x[:, 4:]
        if multi_label and nc > 1:
            thresholds = per_class_conf.to(scores.device) if per_class_conf is not None else conf_thres
            row, cls = torch.where(scores > thresholds)
            boxes, conf = boxes[row], scores[row, cls]
        else:
            conf, cls = scores.max(1)
            thr = per_class_conf.to(conf.device)[cls] if per_class_conf is not None else conf_thres
            mask = conf > thr
            boxes, conf, cls = boxes[mask], conf[mask], cls[mask]
        if boxes.shape[0] == 0:
            outputs.append(torch.zeros((0, 6), device=prediction.device))
            continue
        if boxes.shape[0] > max_nms:
            top = conf.argsort(descending=True)[:max_nms]
            boxes, conf, cls = boxes[top], conf[top], cls[top]
        # class-aware NMS via coordinate offset trick
        coord_span = (boxes.max() - boxes.min()).clamp(min=0) + 1
        offset = torch.zeros_like(cls) if agnostic else cls.float() * coord_span
        keep = nms(boxes + offset[:, None], conf, iou_thres)[:max_det]
        outputs.append(torch.cat((boxes[keep], conf[keep, None], cls[keep, None].float()), 1))
    _ = nc
    return outputs
