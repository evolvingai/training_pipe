"""Backward-compatible namespace for :mod:`trainer.augmentation`."""

from trainer.augmentation import (
    TRANSFORMS,
    Compose,
    Transform,
    build_pipeline,
    filter_boxes,
    pipeline_from_spec,
    register_transform,
)
from trainer.augmentation import geometric, photometric  # noqa: F401

__all__ = [
    "TRANSFORMS",
    "Compose",
    "Transform",
    "build_pipeline",
    "filter_boxes",
    "pipeline_from_spec",
    "register_transform",
]
