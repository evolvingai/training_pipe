"""Detector architecture aliases exposed through the unified registry."""

from __future__ import annotations

from trainer.detection.nn.model import DetectionModel
from trainer.registry import ArchitecturePlugin, architectures


def _builder(spec: str):
    def build(num_classes: int = 80, **_: object) -> DetectionModel:
        return DetectionModel(spec, nc=num_classes)
    return build


for _variant in ("trainer", "trainer-csp", "trainer-fpn"):
    for _scale in ("n", "s", "m", "l", "x"):
        _name = f"{_variant}-{_scale}"
        architectures.register(ArchitecturePlugin(
            task="detection", name=_name, builder=_builder(_name),
            description="TOML-defined anchor-free detector architecture.",
            family="Anchor-free detector",
            input_contract="RGB image tensor with class-labelled bounding boxes.",
            output_contract="Multi-scale boxes, objectness and class scores with NMS predictions.",
            training_strategy="Supervised detection with shared geometric, photometric and multi-image augmentation.",
            stages=("image", "backbone", "feature pyramid", "detection heads", "boxes + classes"),
        ))
