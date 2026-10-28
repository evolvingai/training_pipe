"""Foreground teacher/student losses, separate from the inference architecture.

This is an in-house GT-center-mask formulation, not a claimed reproduction of
any published detector-distillation paper. All teacher targets are detached.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from trainer.detection.utils.boxes import xywh2xyxy
from trainer.detection.utils.checkpoints import load_checkpoint, model_from_checkpoint
from trainer.detection.utils.provenance import file_sha256


def foreground_mask(batch, shape, device):
    b, _, h, w = shape
    yy, xx = torch.meshgrid((torch.arange(h, device=device) + .5) / h,
                            (torch.arange(w, device=device) + .5) / w, indexing="ij")
    mask = torch.zeros((b, h, w), device=device, dtype=torch.bool)
    boxes = xywh2xyxy(batch["bboxes"].to(device))
    indices = batch["batch_idx"].to(device)
    for index in range(b):
        current = boxes[indices == index]
        # Bound temporary memory even in heavily mosaicked/crowded images.
        for chunk in current.split(64):
            if chunk.numel():
                inside = ((xx >= chunk[:, 0, None, None]) & (xx <= chunk[:, 2, None, None]) &
                          (yy >= chunk[:, 1, None, None]) & (yy <= chunk[:, 3, None, None]))
                mask[index] |= inside.any(0)
    return mask


class FeatureAdapters(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(s, t, 1, bias=False) for s, t in zip(student_channels, teacher_channels))

    def forward(self, features):
        return [layer(feature) for layer, feature in zip(self.layers, features)]


class Distiller:
    def __init__(self, student, cfg, names, device):
        self.cfg = cfg
        self.teacher_sha256 = file_sha256(cfg["teacher"])
        checkpoint = load_checkpoint(cfg["teacher"])
        if checkpoint.get("names") != names:
            raise ValueError("teacher and student dataset class names/order must match exactly")
        self.teacher = model_from_checkpoint(checkpoint).to(device)
        from trainer.detection.api import _strip_compiled_prefix
        weights = checkpoint.get("model")
        if not weights:
            raise ValueError("teacher requires nonempty trained model/EMA weights")
        self.teacher.load_state_dict(_strip_compiled_prefix(weights))
        self.teacher.eval().requires_grad_(False)
        if self.teacher.nc != student.nc or self.teacher.ch != student.ch:
            raise ValueError("teacher/student class count and input channels must agree")
        if not torch.equal(self.teacher.stride.cpu(), student.stride.cpu()):
            raise ValueError("teacher/student must have the same ordered feature strides")
        if cfg["distill_box_weight"] > 0 and self.teacher.head.reg_max != student.head.reg_max:
            raise ValueError("box-distribution distillation requires matching reg_max")
        self.adapters = (FeatureAdapters(student.head.in_channels, self.teacher.head.in_channels).to(device)
                         if cfg["distill_feature_weight"] > 0 else None)
        self.reg_max, self.nc = student.head.reg_max, student.nc

    def teacher_forward(self, images):
        # eval mode also freezes running statistics; no validation/test samples enter here.
        self.teacher.eval()
        with torch.no_grad():
            return self.teacher(images, return_features=True)

    def loss(self, student_maps, student_features, teacher_output, batch):
        teacher_maps, teacher_features = teacher_output
        projected = self.adapters(student_features) if self.adapters is not None else student_features
        zero = sum(value.float().sum() * 0 for value in student_maps)
        terms = {name: zero for name in ("feature", "cls", "box")}
        temperature = float(self.cfg["distill_temperature"])
        for raw, target, feature, teacher_feature in zip(student_maps, teacher_maps, projected, teacher_features):
            if raw.shape[-2:] != target.shape[-2:] or feature.shape[-2:] != teacher_feature.shape[-2:]:
                raise ValueError("teacher/student feature grids must align on the same augmented input")
            mask = foreground_mask(batch, raw.shape, raw.device).float()
            denominator = mask.sum().clamp_min(1)
            if self.cfg["distill_feature_weight"] > 0:
                # Channel normalization makes this robust to teacher feature scale.
                error = (F.normalize(feature.float(), dim=1) - F.normalize(teacher_feature.detach().float(), dim=1)).square().sum(1)
                terms["feature"] = terms["feature"] + (error * mask).sum() / denominator
            if self.cfg["distill_cls_weight"] > 0:
                student = raw[:, -self.nc:].float() / temperature
                teacher = target[:, -self.nc:].detach().float() / temperature
                probabilities = teacher.sigmoid()
                kl = (probabilities * (F.logsigmoid(teacher) - F.logsigmoid(student)) +
                      (1 - probabilities) * (F.logsigmoid(-teacher) - F.logsigmoid(-student))).mean(1)
                terms["cls"] = terms["cls"] + (kl * mask).sum() * temperature**2 / denominator
            if self.cfg["distill_box_weight"] > 0:
                b, _, h, w = raw.shape
                student = raw[:, :4*self.reg_max].float().reshape(b, 4, self.reg_max, h, w) / temperature
                teacher = target[:, :4*self.reg_max].detach().float().reshape(b, 4, self.reg_max, h, w) / temperature
                kl = F.kl_div(student.log_softmax(2), teacher.softmax(2), reduction="none").sum(2).mean(1)
                terms["box"] = terms["box"] + (kl * mask).sum() * temperature**2 / denominator
        terms = {name: value / len(student_maps) for name, value in terms.items()}
        total = sum(self.cfg[f"distill_{name}_weight"] * value for name, value in terms.items())
        return total * student_maps[0].shape[0], {name: value.detach() for name, value in terms.items()}

    def state_dict(self):
        adapters = self.adapters.module if hasattr(self.adapters, "module") else self.adapters
        return {"teacher_sha256": self.teacher_sha256,
                "adapters": adapters.state_dict() if adapters is not None else None}

    def load_state_dict(self, saved):
        if saved is None or saved.get("teacher_sha256") != self.teacher_sha256:
            raise ValueError("distillation teacher changed or checkpoint lacks distillation state")
        adapters = self.adapters.module if hasattr(self.adapters, "module") else self.adapters
        if (adapters is None) != (saved["adapters"] is None):
            raise ValueError("distillation adapter structure changed")
        if adapters is not None:
            adapters.load_state_dict(saved["adapters"])
