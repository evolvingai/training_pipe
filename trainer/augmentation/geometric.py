"""Geometric, label-aware transforms built on OpenCV."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from trainer.augmentation.registry import Transform, filter_boxes, register_transform


def _combined_target(samples: list[dict], weights: list[float] | None = None) -> np.ndarray | None:
    """Combine classification targets while leaving detection labels untouched."""
    targets = [np.asarray(sample["target"], dtype=np.float32) for sample in samples if "target" in sample]
    if len(targets) != len(samples):
        return None
    values = np.asarray(weights if weights is not None else [1.0] * len(samples), dtype=np.float32)
    values /= max(float(values.sum()), 1e-9)
    return sum(target * weight for target, weight in zip(targets, values, strict=True)).astype(np.float32)


@register_transform
class LetterBox(Transform):
    """Resize with preserved aspect ratio and pad to a square canvas.

    Stores ``ratio`` and ``pad`` in the sample for coordinate back-mapping.
    """

    def __init__(self, imgsz: int = 640, pad_value: int = 114, center: bool = True):
        self.imgsz = imgsz
        self.pad_value = pad_value
        self.center = center

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        img = sample["image"]
        h, w = img.shape[:2]

        r = min(self.imgsz / h, self.imgsz / w)
        new_w, new_h = round(w * r), round(h * r)
        if (new_w, new_h) != (w, h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw, dh = self.imgsz - new_w, self.imgsz - new_h
        if self.center:
            left, top = dw // 2, dh // 2
        else:
            left, top = 0, 0
        right, bottom = dw - left, dh - top
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(self.pad_value,) * 3)
        boxes = sample["boxes"]
        if boxes.shape[0]:
            boxes = boxes * r
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
        sample.update(image=img, boxes=boxes, ratio=r, pad=(left, top))
        return sample


@register_transform
class RandomFlip(Transform):
    def __init__(self, direction: str = "horizontal", p: float = 0.5):
        assert direction in ("horizontal", "vertical")
        self.direction = direction
        self.p = p

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        if rng.random() >= self.p:
            return sample
        img = sample["image"]
        h, w = img.shape[:2]
        boxes = sample["boxes"]
        if self.direction == "horizontal":
            sample["image"] = np.ascontiguousarray(img[:, ::-1])
            if boxes.shape[0]:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        else:
            sample["image"] = np.ascontiguousarray(img[::-1])
            if boxes.shape[0]:
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        sample["boxes"] = boxes
        return sample


@register_transform
class RandomPerspective(Transform):
    """Random affine/perspective warp: rotation, translation, scale, shear, perspective."""

    def __init__(self, degrees: float = 0.0, translate: float = 0.1, scale: float = 0.5,
                 shear: float = 0.0, perspective: float = 0.0, pad_value: int = 114,
                 min_visibility: float = 0.1):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.pad_value = pad_value
        self.min_visibility = min_visibility

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        img = sample["image"]
        h, w = img.shape[:2]

        # Mosaic emits a large native-scale canvas and requests a crop in the
        # same warp. This keeps object scale aligned with validation images.
        border = sample.pop("mosaic_border", (0, 0))
        out_w = w + border[1] * 2
        out_h = h + border[0] * 2

        center = np.eye(3)
        center[0, 2] = -w / 2
        center[1, 2] = -h / 2

        persp = np.eye(3)
        persp[2, 0] = rng.uniform(-self.perspective, self.perspective)
        persp[2, 1] = rng.uniform(-self.perspective, self.perspective)

        rot = np.eye(3)
        angle = rng.uniform(-self.degrees, self.degrees)
        scale = rng.uniform(1 - self.scale, 1 + self.scale)
        rot[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale)

        shear_m = np.eye(3)
        shear_m[0, 1] = math.tan(rng.uniform(-self.shear, self.shear) * math.pi / 180)
        shear_m[1, 0] = math.tan(rng.uniform(-self.shear, self.shear) * math.pi / 180)

        trans = np.eye(3)
        trans[0, 2] = rng.uniform(0.5 - self.translate, 0.5 + self.translate) * out_w
        trans[1, 2] = rng.uniform(0.5 - self.translate, 0.5 + self.translate) * out_h

        m = trans @ shear_m @ rot @ persp @ center
        if self.perspective:
            img = cv2.warpPerspective(img, m, (out_w, out_h), borderValue=(self.pad_value,) * 3)
        else:
            img = cv2.warpAffine(img, m[:2], (out_w, out_h), borderValue=(self.pad_value,) * 3)

        boxes = sample["boxes"]
        if boxes.shape[0]:
            original_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (scale**2)
            n = boxes.shape[0]
            corners = np.ones((n * 4, 3))
            corners[:, :2] = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(n * 4, 2)
            corners = corners @ m.T
            if self.perspective:
                corners = corners[:, :2] / corners[:, 2:3]
            else:
                corners = corners[:, :2]
            corners = corners.reshape(n, 8)
            xs, ys = corners[:, 0::2], corners[:, 1::2]
            boxes = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], 1).astype(np.float32)
            sample["image"] = img
            sample["boxes"] = boxes
            return filter_boxes(sample, original_areas=original_areas.astype(np.float32),
                                min_visibility=self.min_visibility)
        sample["image"] = img
        return sample


@register_transform
class SquareCrop(Transform):
    """Largest-square crop with optional jitter along the long axis (no distortion)."""

    def __init__(self, jitter: float = 0.0, min_visibility: float = 0.1):
        self.jitter = jitter
        self.min_visibility = min_visibility

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        img = sample["image"]
        h, w = img.shape[:2]
        side = min(h, w)
        if h == w:
            return sample
        max_off = (max(h, w) - side)
        base = max_off / 2
        off = int(base + rng.uniform(-self.jitter, self.jitter) * max_off)
        off = max(0, min(off, max_off))
        boxes = sample["boxes"]
        areas = None
        if boxes.shape[0]:
            areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).astype(np.float32)
        if w > h:
            sample["image"] = np.ascontiguousarray(img[:, off:off + side])
            if boxes.shape[0]:
                boxes[:, [0, 2]] -= off
        else:
            sample["image"] = np.ascontiguousarray(img[off:off + side])
            if boxes.shape[0]:
                boxes[:, [1, 3]] -= off
        sample["boxes"] = boxes
        return filter_boxes(sample, original_areas=areas, min_visibility=self.min_visibility)


@register_transform
class Mosaic(Transform):
    """4- or 9-tile mosaic composition. Requires dataset access."""

    requires_dataset = True

    def __init__(self, imgsz: int = 640, p: float = 1.0, p9: float = 0.0, pad_value: int = 114):
        self.imgsz = imgsz
        self.p = p
        self.p9 = p9
        self.pad_value = pad_value

    def _load(self, idx: int) -> dict:
        return self.dataset.load_sample(idx)

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        if rng.random() >= self.p:
            return sample
        n = 9 if rng.random() < self.p9 else 4
        sample = self._mosaic4(sample, rng) if n == 4 else self._mosaic9(sample, rng)
        sample["mosaic_border"] = (-self.imgsz // 2, -self.imgsz // 2)
        return sample

    def _mosaic4(self, sample: dict, rng: random.Random) -> dict:
        s = self.imgsz
        canvas = np.full((s * 2, s * 2, 3), self.pad_value, dtype=np.uint8)
        cx = int(rng.uniform(s * 0.5, s * 1.5))
        cy = int(rng.uniform(s * 0.5, s * 1.5))
        samples = [sample] + [self._load(rng.randrange(len(self.dataset))) for _ in range(3)]
        all_boxes, all_cls, target_weights = [], [], []
        for i, sm in enumerate(samples):
            img, boxes, cls = sm["image"], sm["boxes"], sm["cls"]
            h, w = img.shape[:2]
            r = min(s / h, s / w)
            img = cv2.resize(img, (round(w * r), round(h * r)))
            h, w = img.shape[:2]
            if i == 0:
                x1, y1, x2, y2 = max(cx - w, 0), max(cy - h, 0), cx, cy
                sx1, sy1 = w - (x2 - x1), h - (y2 - y1)
            elif i == 1:
                x1, y1, x2, y2 = cx, max(cy - h, 0), min(cx + w, s * 2), cy
                sx1, sy1 = 0, h - (y2 - y1)
            elif i == 2:
                x1, y1, x2, y2 = max(cx - w, 0), cy, cx, min(cy + h, s * 2)
                sx1, sy1 = w - (x2 - x1), 0
            else:
                x1, y1, x2, y2 = cx, cy, min(cx + w, s * 2), min(cy + h, s * 2)
                sx1, sy1 = 0, 0
            canvas[y1:y2, x1:x2] = img[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
            target_weights.append(float((x2 - x1) * (y2 - y1)))
            if boxes.shape[0]:
                b = boxes * r
                b[:, [0, 2]] += x1 - sx1
                b[:, [1, 3]] += y1 - sy1
                all_boxes.append(b)
                all_cls.append(cls)
        sample["image"] = canvas
        target = _combined_target(samples, target_weights)
        if target is not None:
            sample["target"] = target
        sample["boxes"] = np.concatenate(all_boxes, 0).astype(np.float32) if all_boxes else np.zeros((0, 4), np.float32)
        sample["cls"] = np.concatenate(all_cls, 0).astype(np.float32) if all_cls else np.zeros((0,), np.float32)
        return filter_boxes(sample)

    def _mosaic9(self, sample: dict, rng: random.Random) -> dict:
        s = self.imgsz
        canvas = np.full((s * 3, s * 3, 3), self.pad_value, dtype=np.uint8)
        samples = [sample] + [self._load(rng.randrange(len(self.dataset))) for _ in range(8)]
        all_boxes, all_cls = [], []
        for i, sm in enumerate(samples):
            img, boxes, cls = sm["image"], sm["boxes"], sm["cls"]
            h, w = img.shape[:2]
            r = min(s / h, s / w)
            img = cv2.resize(img, (round(w * r), round(h * r)))
            h, w = img.shape[:2]
            gr, gc = divmod(i, 3)
            x1, y1 = gc * s, gr * s
            canvas[y1:y1 + h, x1:x1 + w] = img
            if boxes.shape[0]:
                b = boxes * r
                b[:, [0, 2]] += x1
                b[:, [1, 3]] += y1
                all_boxes.append(b)
                all_cls.append(cls)
        # center-crop 3s canvas down to 2s so downstream scale matches mosaic4
        off = s // 2
        canvas = canvas[off:off + 2 * s, off:off + 2 * s]
        boxes = np.concatenate(all_boxes, 0).astype(np.float32) if all_boxes else np.zeros((0, 4), np.float32)
        if boxes.shape[0]:
            boxes[:, [0, 2]] -= off
            boxes[:, [1, 3]] -= off
        sample["image"] = canvas
        target = _combined_target(samples)
        if target is not None:
            sample["target"] = target
        sample["boxes"] = boxes
        sample["cls"] = np.concatenate(all_cls, 0).astype(np.float32) if all_cls else np.zeros((0,), np.float32)
        return filter_boxes(sample)


@register_transform
class MixUp(Transform):
    """Blend two samples with a Beta(alpha, alpha) ratio; labels are unioned."""

    requires_dataset = True

    def __init__(self, p: float = 0.0, alpha: float = 1.5):
        self.p = p
        self.alpha = alpha

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        if rng.random() >= self.p:
            return sample
        other = self.dataset.load_sample(rng.randrange(len(self.dataset)))
        img1, img2 = sample["image"], other["image"]
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])

        def pad_to(img: np.ndarray) -> np.ndarray:
            out = np.full((h, w, 3), 114, dtype=np.uint8)
            out[: img.shape[0], : img.shape[1]] = img
            return out

        lam = np.random.default_rng(rng.randrange(2**31)).beta(self.alpha, self.alpha)
        blended = (pad_to(img1).astype(np.float32) * lam + pad_to(img2).astype(np.float32) * (1 - lam))
        sample["image"] = blended.astype(np.uint8)
        target = _combined_target([sample, other], [float(lam), float(1 - lam)])
        if target is not None:
            sample["target"] = target
        sample["boxes"] = np.concatenate([sample["boxes"], other["boxes"]], 0).astype(np.float32)
        sample["cls"] = np.concatenate([sample["cls"], other["cls"]], 0).astype(np.float32)
        return sample


@register_transform
class CopyPaste(Transform):
    """IoU-constrained copy-paste: paste object crops from another image so they
    overlap existing boxes within a target IoU band (occlusion simulation)."""

    requires_dataset = True

    def __init__(self, p: float = 0.0, iou_range: tuple[float, float] = (0.10, 0.55), max_paste: int = 3,
                 attempts: int = 10):
        self.p = p
        self.iou_range = iou_range
        self.max_paste = max_paste
        self.attempts = attempts

    @staticmethod
    def _iou_one(box: np.ndarray, others: np.ndarray) -> np.ndarray:
        if others.shape[0] == 0:
            return np.zeros(0)
        x1 = np.maximum(box[0], others[:, 0])
        y1 = np.maximum(box[1], others[:, 1])
        x2 = np.minimum(box[2], others[:, 2])
        y2 = np.minimum(box[3], others[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = (box[2] - box[0]) * (box[3] - box[1])
        a2 = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
        return inter / (a1 + a2 - inter + 1e-9)

    def __call__(self, sample: dict, rng: random.Random) -> dict:
        if rng.random() >= self.p:
            return sample
        donor = self.dataset.load_sample(rng.randrange(len(self.dataset)))
        if "target" in sample and sample["boxes"].shape[0] == 0:
            return self._classification_paste(sample, donor, rng)
        if sample["boxes"].shape[0] == 0:
            return sample
        if donor["boxes"].shape[0] == 0:
            return sample
        img = sample["image"]
        h, w = img.shape[:2]
        lo, hi = self.iou_range
        n_paste = rng.randint(1, self.max_paste)
        order = list(range(donor["boxes"].shape[0]))
        rng.shuffle(order)
        for di in order[:n_paste]:
            db = donor["boxes"][di].astype(int)
            crop = donor["image"][db[1]:db[3], db[0]:db[2]]
            ch, cw = crop.shape[:2]
            if ch < 4 or cw < 4 or ch >= h or cw >= w:
                continue
            for _ in range(self.attempts):
                px = rng.randint(0, w - cw - 1)
                py = rng.randint(0, h - ch - 1)
                new_box = np.array([px, py, px + cw, py + ch], dtype=np.float32)
                ious = self._iou_one(new_box, sample["boxes"])
                if ious.size and lo <= float(ious.max()) <= hi:
                    img[py:py + ch, px:px + cw] = crop
                    sample["boxes"] = np.concatenate([sample["boxes"], new_box[None]], 0)
                    sample["cls"] = np.concatenate([sample["cls"], donor["cls"][di:di + 1]], 0)
                    break
        sample["image"] = img
        return sample

    def _classification_paste(self, sample: dict, donor: dict, rng: random.Random) -> dict:
        """Use a CutMix-style region when object boxes are unavailable."""
        image = sample["image"].copy()
        height, width = image.shape[:2]
        donor_image = cv2.resize(donor["image"], (width, height), interpolation=cv2.INTER_LINEAR)
        lo, hi = self.iou_range
        fraction = rng.uniform(max(0.01, lo), min(0.95, hi))
        aspect = rng.uniform(0.5, 2.0)
        patch_height = max(1, min(height, int((fraction * height * width * aspect) ** 0.5)))
        patch_width = max(1, min(width, int((fraction * height * width / aspect) ** 0.5)))
        x = rng.randint(0, max(width - patch_width, 0))
        y = rng.randint(0, max(height - patch_height, 0))
        image[y:y + patch_height, x:x + patch_width] = donor_image[y:y + patch_height, x:x + patch_width]
        actual = patch_height * patch_width / max(height * width, 1)
        target = _combined_target([sample, donor], [1.0 - actual, actual])
        sample["image"] = image
        if target is not None:
            sample["target"] = target
        return sample
