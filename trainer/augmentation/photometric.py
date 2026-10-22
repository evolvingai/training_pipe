"""Photometric / sensor-robustness transforms (boxes untouched)."""

from __future__ import annotations

import random

import cv2
import numpy as np

from trainer.augmentation.registry import Transform, register_transform


@register_transform
class RandomHSV(Transform):
    def __init__(self, h: float = 0.015, s: float = 0.7, v: float = 0.4, p: float = 1.0):
        self.h, self.s, self.v = h, s, v
        self.p = p

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        if rng.random() >= self.p or (self.h == 0 and self.s == 0 and self.v == 0):
            return sample
        gains = np.array([rng.uniform(-1, 1) * self.h, rng.uniform(-1, 1) * self.s,
                          rng.uniform(-1, 1) * self.v]) + 1
        hsv = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        x = np.arange(256)
        lut_h = ((x * gains[0]) % 180).astype(np.uint8)
        lut_s = np.clip(x * gains[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(x * gains[2], 0, 255).astype(np.uint8)
        hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
        sample["image"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return sample


@register_transform
class BrightnessContrast(Transform):
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        if rng.random() >= self.p:
            return sample
        alpha = 1.0 + rng.uniform(-self.contrast, self.contrast)
        beta = 255 * rng.uniform(-self.brightness, self.brightness)
        sample["image"] = cv2.convertScaleAbs(sample["image"], alpha=alpha, beta=beta)
        return sample


@register_transform
class MotionBlur(Transform):
    def __init__(self, p: float = 0.0, max_kernel: int = 9):
        self.p = p
        self.max_kernel = max_kernel

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        if rng.random() >= self.p:
            return sample
        k = rng.randrange(3, self.max_kernel + 1, 2)
        kernel = np.zeros((k, k), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[k // 2, :] = 1.0
        else:
            kernel[:, k // 2] = 1.0
        kernel /= kernel.sum()
        sample["image"] = cv2.filter2D(sample["image"], -1, kernel)
        return sample


@register_transform
class GaussNoise(Transform):
    def __init__(self, p: float = 0.0, sigma_max: float = 12.0):
        self.p = p
        self.sigma_max = sigma_max

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        if rng.random() >= self.p:
            return sample
        sigma = rng.uniform(2.0, self.sigma_max)
        noise = np.random.default_rng(rng.randrange(2**31)).normal(0, sigma, sample["image"].shape)
        sample["image"] = np.clip(sample["image"].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return sample


@register_transform
class JPEGArtifacts(Transform):
    def __init__(self, p: float = 0.0, quality_range: tuple[int, int] = (30, 80)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        if rng.random() >= self.p:
            return sample
        quality = rng.randint(*self.quality_range)
        ok, enc = cv2.imencode(".jpg", sample["image"], [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            sample["image"] = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return sample


@register_transform
class RandomErasing(Transform):
    """Occlusion simulation: fill random rectangles with noise (boxes kept)."""

    def __init__(self, p: float = 0.0, area_range: tuple[float, float] = (0.02, 0.1), max_patches: int = 3):
        self.p = p
        self.area_range = area_range
        self.max_patches = max_patches

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        if rng.random() >= self.p:
            return sample
        img = sample["image"]
        h, w = img.shape[:2]
        for _ in range(rng.randint(1, self.max_patches)):
            area = rng.uniform(*self.area_range) * h * w
            aspect = rng.uniform(0.5, 2.0)
            ph = int((area * aspect) ** 0.5)
            pw = int((area / aspect) ** 0.5)
            if ph >= h or pw >= w or ph < 2 or pw < 2:
                continue
            py = rng.randint(0, h - ph - 1)
            px = rng.randint(0, w - pw - 1)
            patch = np.random.default_rng(rng.randrange(2**31)).integers(0, 256, (ph, pw, 3), dtype=np.uint8)
            img[py:py + ph, px:px + pw] = patch
        sample["image"] = img
        return sample
