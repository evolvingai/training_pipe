"""Shared image augmentation for detection, classification, video, and VLM tasks."""

from trainer.augmentation.registry import (
    TRANSFORMS,
    Compose,
    Transform,
    build_pipeline,
    filter_boxes,
    pipeline_from_spec,
    register_transform,
)
from trainer.augmentation.config import (
    AUGMENTATION_DEFAULTS,
    CLASSIFICATION_AUGMENTATION_DEFAULTS,
    PASSTHROUGH_AUGMENTATION_DEFAULTS,
    add_augmentation_arguments,
    augmentation_config,
    augmentation_overrides,
    load_augmentation_config,
)
from trainer.augmentation import geometric, photometric  # noqa: F401  (populate registry)

__all__ = [
    "TRANSFORMS",
    "Compose",
    "Transform",
    "build_pipeline",
    "filter_boxes",
    "pipeline_from_spec",
    "register_transform",
    "AUGMENTATION_DEFAULTS",
    "CLASSIFICATION_AUGMENTATION_DEFAULTS",
    "PASSTHROUGH_AUGMENTATION_DEFAULTS",
    "add_augmentation_arguments",
    "augmentation_config",
    "augmentation_overrides",
    "load_augmentation_config",
]
