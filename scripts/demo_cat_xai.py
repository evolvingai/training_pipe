#!/usr/bin/env python3
"""Run ImageNet ConvNeXt Tiny on a cat photo and export Trainer xAI artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from trainer.classification.data import select_device
from trainer.classification.xai import generate_xai


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--image", type=Path, required=True)
    root.add_argument("--output", type=Path, required=True)
    root.add_argument("--device", default="auto")
    root.add_argument("--target", default="predicted")
    root.add_argument("--intensity", type=float, default=0.72)
    root.add_argument("--activation-channels", type=int, default=4)
    return root


def main() -> int:
    args = parser().parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    device = select_device(args.device)
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights).to(device).eval()
    transform = weights.transforms()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def preprocess(path: Path) -> tuple[torch.Tensor, np.ndarray]:
        with Image.open(path) as source:
            tensor = transform(source.convert("RGB"))
        rgb = np.clip(((tensor.cpu() * std + mean).permute(1, 2, 0).numpy()) * 255.0, 0, 255).astype(np.uint8)
        return tensor, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    result = generate_xai(
        model,
        {
            "classes": list(weights.meta["categories"]),
            "arch": "torchvision-convnext-tiny-imagenet1k",
            "imgsz": 224,
            "preprocessing": "ImageNet resize 236 | center crop 224 | official normalisation",
        },
        [args.image], args.output, device, target=args.target, max_images=1,
        intensity=args.intensity, activation_channels=args.activation_channels,
        preprocess=preprocess,
    )
    item = result["images"][0]
    print(json.dumps({
        "predicted": item["predicted"], "confidence": item["confidence"],
        "target": item["target"], "dossier": str(args.output / item["dossier"]),
        "gallery": result["gallery"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
