"""Runtime checks: device selection, seeding, autobatch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from trainer.detection.utils.logging import LOGGER


def select_device(device: str = "") -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_seeds(seed: int = 0, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def autobatch(model: torch.nn.Module, imgsz: int, device: torch.device, amp: bool = True,
              start: int = 2, max_batch: int = 512, mem_fraction: float = 0.85) -> int:
    """Find the largest safe batch size by doubling until OOM, then back off to 85%."""
    if device.type != "cuda":
        LOGGER.warning("autobatch requires CUDA; falling back to batch=16")
        return 16
    model = model.to(device).train()
    batch = start
    last_ok = start
    while batch <= max_batch:
        try:
            imgs = torch.zeros((batch, 3, imgsz, imgsz), device=device)
            with torch.autocast(device_type="cuda", enabled=amp):
                out = model(imgs)
                loss = sum(o.sum() for o in out) if isinstance(out, (list, tuple)) else out.sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            last_ok = batch
            batch *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    selected = max(int(last_ok * mem_fraction), 1)
    LOGGER.info(f"autobatch: selected batch={selected} (largest OK: {last_ok})")
    return selected
