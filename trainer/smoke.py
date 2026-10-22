"""Synthetic one-step smoke tests for architecture plugins."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from trainer.registry import architectures, build_architecture
from trainer.tasks import resolve_task_name

DEFAULT_SMOKE_ARCH = {
    "classification": "convnext-tiny",
    "video-classification": "r2plus1d-tiny",
    "vlm": "tiny-vlm-lora",
}


def _optimize(model: torch.nn.Module, loss_fn) -> tuple[float, int]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("architecture exposes no trainable parameters")
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn()
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite smoke loss: {float(loss)}")
    loss.backward()
    optimizer.step()
    changed = sum(not torch.equal(old, new.detach()) for old, new in zip(before, trainable, strict=True))
    if not changed:
        raise RuntimeError("optimizer step did not update any trainable parameter")
    return float(loss.detach()), changed


def smoke_architecture(task: str, arch: str | None = None, device: str = "cpu") -> dict[str, Any]:
    canonical = resolve_task_name(task)
    selected = arch or DEFAULT_SMOKE_ARCH.get(canonical)
    if not selected:
        raise ValueError(f"no synthetic smoke fixture for task '{task}'")
    target = torch.device(device)
    torch.manual_seed(7)
    if canonical == "classification":
        model = build_architecture(canonical, selected, num_classes=3, width=8, dropout=0.0).to(target)
        images, labels = torch.randn(2, 3, 32, 32, device=target), torch.tensor([0, 2], device=target)
        loss, changed = _optimize(model, lambda: F.cross_entropy(model(images), labels))
        output_shape = list(model(images).shape)
    elif canonical == "video-classification":
        model = build_architecture(canonical, selected, num_classes=3, width=4, dropout=0.0).to(target)
        clips, labels = torch.randn(2, 4, 3, 24, 24, device=target), torch.tensor([0, 2], device=target)
        loss, changed = _optimize(model, lambda: F.cross_entropy(model(clips), labels))
        output_shape = list(model(clips).shape)
    elif canonical == "vlm":
        model = build_architecture(canonical, selected, vocab_size=32, embed_dim=16, rank=2).to(target)
        images = torch.randn(2, 3, 16, 16, device=target)
        tokens = torch.randint(0, 32, (2, 6), device=target)
        labels = torch.randint(0, 32, (2, 6), device=target)
        loss, changed = _optimize(model, lambda: F.cross_entropy(model(images, tokens).flatten(0, 1), labels.flatten()))
        output_shape = list(model(images, tokens).shape)
    else:
        raise ValueError("detection smoke remains available through scripts/smoke_e2e.py")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"task": canonical, "architecture": architectures.get(canonical, selected).name, "device": str(target),
            "loss": loss, "parameters": parameters, "trainable_parameters": trainable,
            "updated_tensors": changed, "output_shape": output_shape, "ok": math.isfinite(loss)}


def smoke_all(device: str = "cpu") -> list[dict[str, Any]]:
    return [smoke_architecture(task, arch, device) for task, arch in DEFAULT_SMOKE_ARCH.items()]
