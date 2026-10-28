"""Inference: preprocess, forward, NMS, coordinate back-mapping, Results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from trainer.augmentation.geometric import LetterBox
from trainer.detection.data.dataset import IMG_SUFFIXES
from trainer.detection.utils.boxes import scale_boxes
from trainer.detection.utils.image import imread_rgb
from trainer.detection.utils.logging import LOGGER
from trainer.detection.utils.nms import non_max_suppression
from trainer.detection.utils.plotting import draw_boxes


@dataclass
class Results:
    path: str
    orig_shape: tuple[int, int]
    boxes: np.ndarray          # (n, 4) xyxy in original image pixels
    conf: np.ndarray           # (n,)
    cls: np.ndarray            # (n,) int
    names: dict[int, str] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.conf)

    def plot(self, image: np.ndarray | None = None) -> np.ndarray:
        """Draw the detections on the source image. Takes and returns RGB."""
        img = image if image is not None else imread_rgb(self.path)
        return draw_boxes(img, self.boxes, self.cls, self.names, self.conf)

    def to_label_txt(self) -> str:
        """Predictions in training label format: cls cx cy w h (normalized)."""
        h, w = self.orig_shape
        lines = []
        for box, c in zip(self.boxes, self.cls):
            cx = (box[0] + box[2]) / 2 / w
            cy = (box[1] + box[3]) / 2 / h
            bw = (box[2] - box[0]) / w
            bh = (box[3] - box[1]) / h
            lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "path": self.path,
            "n": len(self),
            "max_conf": float(self.conf.max()) if len(self) else 0.0,
            "mean_conf": float(self.conf.mean()) if len(self) else 0.0,
        }


def gather_sources(source: str | Path | list) -> list[str]:
    if isinstance(source, (list, tuple)):
        out: list[str] = []
        for s in source:
            out.extend(gather_sources(s))
        return out
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"prediction source not found: {p}")
    if p.is_dir():
        paths = sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMG_SUFFIXES)
        if not paths:
            raise FileNotFoundError(f"no supported images found in: {p}")
        return paths
    if p.suffix == ".txt":
        paths = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            item = Path(line.strip())
            paths.append(str(item if item.is_absolute() else p.parent / item))
        if not paths:
            raise FileNotFoundError(f"prediction source list is empty: {p}")
        missing = [item for item in paths if not Path(item).exists()]
        if missing:
            raise FileNotFoundError(f"prediction source list contains missing path: {missing[0]}")
        return paths
    if p.suffix.lower() not in IMG_SUFFIXES:
        raise ValueError(f"unsupported prediction source type: {p}")
    return [str(p)]


class Predictor:
    def __init__(self, model, cfg: dict, names: dict[int, str] | None = None, device=None):
        self.model = model.eval()
        self.cfg = cfg
        self.device = device or next(model.parameters()).device
        self.names = names or getattr(model, "names", {})
        self.letterbox = LetterBox(int(cfg["imgsz"]))
        self.per_class_conf: torch.Tensor | None = None

    def set_per_class_thresholds(self, thresholds: dict[str, float]) -> None:
        """Map {class_name: conf} to a per-class threshold tensor."""
        nc = len(self.names)
        default = float(self.cfg["conf"])
        vec = torch.full((nc,), default)
        by_name = {v: k for k, v in self.names.items()}
        for name, thr in thresholds.items():
            if not 0 <= float(thr) <= 1:
                raise ValueError(f"threshold for class '{name}' must be in [0, 1]")
            if name in by_name:
                vec[by_name[name]] = float(thr)
        self.per_class_conf = vec

    def _preprocess(self, img: np.ndarray) -> tuple[torch.Tensor, float, tuple[float, float]]:
        import random
        sample = {"image": img, "boxes": np.zeros((0, 4), np.float32), "cls": np.zeros(0, np.float32)}
        sample = self.letterbox(sample, random.Random(0))
        tensor = torch.from_numpy(np.ascontiguousarray(sample["image"].transpose(2, 0, 1)))
        return tensor, sample["ratio"], sample["pad"]

    @torch.no_grad()
    def predict(self, source, conf: float | None = None, embed: bool = False,
                batch_size: int = 16) -> list[Results]:
        conf = conf if conf is not None else float(self.cfg["conf"])
        if not 0 <= float(conf) <= 1:
            raise ValueError("confidence threshold must be in [0, 1]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        paths = gather_sources(source)
        total = len(paths)
        LOGGER.info(f"scanning source → {total} image(s) found")
        results: list[Results] = []
        for start in range(0, total, batch_size):
            chunk = paths[start:start + batch_size]
            remaining = total - start - len(chunk)
            LOGGER.info(f"  [{start + 1}–{start + len(chunk)}/{total}] {remaining} remaining")
            imgs, metas = [], []
            for p in chunk:
                orig = imread_rgb(p)
                if orig is None:
                    LOGGER.warning(f"skipping unreadable prediction image: {p}")
                    continue
                tensor, ratio, pad = self._preprocess(orig)
                imgs.append(tensor)
                metas.append((p, orig.shape[:2], ratio, pad))
            if not imgs:
                continue
            batch = torch.stack(imgs).to(self.device).float() / 255.0
            decoded, feats = self.model(batch)
            preds = non_max_suppression(
                decoded.float(), conf_thres=conf, iou_thres=float(self.cfg["iou"]),
                max_det=int(self.cfg["max_det"]), per_class_conf=self.per_class_conf,
                multi_label=bool(self.cfg.get("nms_multi_label", False)),
            )
            embeddings = None
            if embed:
                # global-average-pool the finest pyramid level as the embedding
                embeddings = feats[0].mean(dim=(2, 3)).cpu().numpy()
            for bi, (pred, (p, orig_shape, ratio, pad)) in enumerate(zip(preds, metas)):
                boxes = pred[:, :4].cpu()
                if len(boxes):
                    boxes = scale_boxes(boxes, batch.shape[-2:], orig_shape, pad=pad, gain=ratio)
                results.append(Results(
                    path=p,
                    orig_shape=orig_shape,
                    boxes=boxes.numpy() if len(boxes) else np.zeros((0, 4), np.float32),
                    conf=pred[:, 4].cpu().numpy(),
                    cls=pred[:, 5].cpu().numpy().astype(int),
                    names=self.names,
                    embedding=embeddings[bi] if embeddings is not None else None,
                ))
        return results
