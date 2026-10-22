"""Trainer: unified detection, classification, video, and VLM training."""

from trainer.detection.api import Model
from trainer.api import Trainer
from trainer.registry import build_architecture, register_architecture

TrainerDetector = Model

__all__ = ["Model", "Trainer", "TrainerDetector", "build_architecture", "register_architecture"]
