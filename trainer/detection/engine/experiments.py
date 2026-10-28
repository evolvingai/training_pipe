"""Opt-in experiment routing and synchronized timing; baseline policy is unchanged."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

import torch
from torch import nn

from trainer.detection.nn.modules.attention import AreaAttention, PaperAreaAttention
from trainer.detection.utils.checks import cuda_device_index
from trainer.detection.utils.ema import ModelEMA


def optimizer_partition(model, cfg, extra=None):
    raw = ModelEMA._unwrap(model)
    entries = [("", raw)] + ([("distill.", extra)] if extra is not None else [])
    names = {id(p): prefix + name for prefix, module in entries for name, p in module.named_parameters()}
    outputs = {id(branch[-1].weight) for branch in (*raw.head.reg_convs, *raw.head.cls_convs)}
    stem = next(m for m in raw.modules() if isinstance(m, nn.Conv2d))
    qkv = {
        id(m.qkv.conv.weight)
        for m in raw.modules()
        if isinstance(m, (AreaAttention, PaperAreaAttention))
    }
    biases, no_decay, decay, excluded, split_qkv = [], [], [], [], []
    manifest, seen = [], set()
    for prefix, owner in entries:
        for module in owner.modules():
            for name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if id(p) in seen:
                    raise ValueError(f"parameter registered more than once: {names[id(p)]}")
                seen.add(id(p))
                reason = "matrix"
                if name == "bias":
                    target, reason = biases, "bias"
                elif isinstance(module, nn.BatchNorm2d) or p.ndim == 1:
                    target, reason = no_decay, "normalization/gain"
                else:
                    is_depthwise = isinstance(module, nn.Conv2d) and module.groups == module.in_channels > 1
                    reasons = [label for condition, label in (
                        (bool(prefix), "training_adapter"),
                        (cfg.get("muon_exclude_outputs") and id(p) in outputs, "output"),
                        (cfg.get("muon_exclude_stem") and p is stem.weight, "stem"),
                        (cfg.get("muon_exclude_depthwise") and is_depthwise, "depthwise"),
                        (p.ndim not in (2, 4), "non_matrix"),
                    ) if condition]
                    target = excluded if reasons else split_qkv if cfg.get("muon_split_qkv") and id(p) in qkv else decay
                    reason = "+".join(reasons) or ("qkv_slices" if target is split_qkv else "matrix")
                target.append(p)
                use_muon = cfg["optimizer"] == "musgd" and (target is decay or target is split_qkv)
                manifest.append({"name": names[id(p)], "shape": list(p.shape), "numel": p.numel(),
                                 "route": "muon+sgd" if use_muon else cfg["optimizer"] if cfg["optimizer"] != "musgd" else "sgd",
                                 "reason": reason, "weight_decay": 0. if target is biases or target is no_decay else cfg["weight_decay"],
                                 "qkv_splits": 3 if use_muon and target is split_qkv else 1})
    expected = {id(p) for _, module in entries for p in module.parameters() if p.requires_grad}
    if seen != expected:
        raise AssertionError("optimizer routing must cover each trainable parameter exactly once")
    return biases, no_decay, decay, excluded, split_qkv, names, manifest


class StepProfiler:
    """CUDA sync wall-time measurements; profiling overhead is explicitly included."""
    def __init__(self, enabled, device):
        self.enabled, self.device = bool(enabled), device
        self.values = {}

    def sync(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(cuda_device_index(self.device))

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.values[name] = self.values.get(name, 0.) + time.perf_counter() - start

    def reset(self):
        self.values = {}


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
