"""Image-classification training, evaluation, and deployment workflows."""

from trainer.classification.data import ImageDataset, Sample, TinyClassifier, discover_training
from trainer.classification.engine import main

__all__ = ["ImageDataset", "Sample", "TinyClassifier", "discover_training", "main"]
