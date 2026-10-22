"""Safe, versioned checkpoint loading and model reconstruction."""

from __future__ import annotations

from pathlib import Path

import torch


CHECKPOINT_VERSION = 2


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load tensor/state-only checkpoints without enabling arbitrary pickle code."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"invalid checkpoint {path}: expected a dictionary")
    return checkpoint


def model_from_checkpoint(checkpoint: dict, nc: int | None = None):
    """Rebuild a DetectionModel, preferring the embedded portable architecture."""
    from trainer.detection.nn.model import DetectionModel

    model_spec = checkpoint.get("model_cfg") or checkpoint.get("model_name")
    if model_spec is None:
        raise KeyError("checkpoint is missing both 'model_cfg' and 'model_name'")
    return DetectionModel(
        model_spec,
        nc=int(nc if nc is not None else checkpoint["nc"]),
        ch=int(checkpoint.get("ch", 3)),
        reg_max=int(checkpoint.get("reg_max", 16)),
    )
