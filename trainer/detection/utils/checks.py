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


def cuda_device_index(device: str | torch.device | int) -> int:
    """Return the integer index required by CUDA management/metric APIs.

    Tensor creation and ``Module.to`` accept a CUDA string or ``torch.device``,
    but some vendor PyTorch builds only accept an integer in APIs such as
    ``reset_peak_memory_stats``.  Keep tensor placement expressive and
    normalize only at that CUDA API boundary.
    """
    if isinstance(device, int):
        if device < 0:
            raise ValueError(f"invalid CUDA device index: {device}")
        return device
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError(f"expected a CUDA device, got {parsed}")
    return torch.cuda.current_device() if parsed.index is None else parsed.index


def initialize_cuda_device(device: str | torch.device | int) -> int:
    """Initialize a fresh process's CUDA context and return its device index.

    Some NVIDIA development wheels reject allocator/metric calls before the
    process has created any CUDA tensor.  Queue workers are intentionally fresh
    subprocesses, so initialize their selected device before resetting metrics.
    """
    index = cuda_device_index(device)
    torch.cuda.set_device(index)
    torch.empty(1, device=torch.device("cuda", index))
    return index


def init_seeds(seed: int = 0, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=not deterministic)
    if deterministic:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))


def autobatch(model: torch.nn.Module, imgsz: int, device: torch.device, amp: bool = True,
              start: int = 2, max_batch: int = 512, mem_fraction: float = 0.80,
              optimizer: str = "adamw", include_ema: bool = True) -> int:
    """Estimate a safe CUDA batch while reserving optimizer and EMA memory.

    The probe honors non-RGB input channels and restores model mode, gradients,
    and BatchNorm buffers, so merely selecting a batch cannot alter training.
    """
    if device.type != "cuda":
        LOGGER.warning("autobatch requires CUDA; falling back to batch=16")
        return 16
    if not 0 < mem_fraction < 1:
        raise ValueError("autobatch mem_fraction must be in (0, 1)")
    original_training = model.training
    bn_state = {
        module: (module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats
    }
    model = model.to(device).train()
    cuda_index = cuda_device_index(device)
    channels = int(getattr(model, "ch", 3))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer_bytes = trainable * (4 if str(optimizer).lower() == "sgd" else 8)
    ema_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) if include_ema else 0
    free_bytes, _ = torch.cuda.mem_get_info(cuda_index)
    budget = max(int(free_bytes * mem_fraction) - optimizer_bytes - ema_bytes, 0)
    batch = start
    last_ok = 0
    bytes_per_image = 0.0
    try:
        while batch <= max_batch:
            imgs = out = loss = None
            try:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                baseline = torch.cuda.memory_allocated(cuda_index)
                torch.cuda.reset_peak_memory_stats(cuda_index)
                imgs = torch.zeros((batch, channels, imgsz, imgsz), device=device)
                with torch.autocast(device_type="cuda", enabled=amp):
                    out = model(imgs)
                    loss = sum(o.sum() for o in out) if isinstance(out, (list, tuple)) else out.sum()
                loss.backward()
                torch.cuda.synchronize(cuda_index)
                peak_delta = max(torch.cuda.max_memory_allocated(cuda_index) - baseline, 1)
                bytes_per_image = max(bytes_per_image, peak_delta / batch)
                last_ok = batch
                del imgs, out, loss
                batch *= 2
            except torch.cuda.OutOfMemoryError:
                imgs = out = loss = None
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break
    finally:
        model.zero_grad(set_to_none=True)
        for module, (mean, var, count) in bn_state.items():
            module.running_mean.copy_(mean)
            module.running_var.copy_(var)
            module.num_batches_tracked.copy_(count)
        model.train(original_training)
        torch.cuda.empty_cache()
    if last_ok == 0:
        raise RuntimeError("autobatch could not fit even the minimum probe batch")
    by_budget = int(budget / bytes_per_image) if bytes_per_image else last_ok
    selected = max(min(last_ok, by_budget, max_batch), 1)
    LOGGER.info(
        f"autobatch: selected batch={selected} (probe={last_ok}, channels={channels}, "
        f"reserved={(optimizer_bytes + ema_bytes) / 2**20:.0f} MiB for optimizer/EMA)"
    )
    return selected
