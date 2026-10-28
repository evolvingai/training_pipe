"""COCO 2017 SOTA fixes.

Standalone implementations for the nine issues marked [fixed in SOTA]
in docs/coco2017_issues.md.

Issues covered (by Action Item number):
    #4  standard maxDets=[1, 10, 100] — coco_evaluate()
    #5  pycocotools for eval       — coco_evaluate()
    #6  Filter iscrowd=1           — filter_crowd()
    #7  Focal loss                 — focal_loss()
    #8  Repeat factor sampler      — repeat_factors(), RepeatFactorSampler
    #10 Label smoothing            — smooth_bce()
    #11 Multi-scale training       — MultiScaleCollate
    #12 Copy-paste augmentation    — trainer.augmentation.geometric.CopyPaste (already implemented)
    #13 Letterbox resizing         — trainer.augmentation.geometric.LetterBox  (already implemented)
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# #6  Filter iscrowd=1
# ---------------------------------------------------------------------------

def filter_crowd(annotations_json: str | Path, out_path: str | Path | None = None) -> dict:
    """Remove iscrowd=1 annotations from a COCO instances JSON.

    Args:
        annotations_json: Path to a COCO instances JSON file.
        out_path:         If given, the filtered JSON is written here.

    Returns:
        Filtered data dict (annotations key contains only iscrowd=0 entries).
    """
    data = json.loads(Path(annotations_json).read_text())
    before = len(data["annotations"])
    data["annotations"] = [a for a in data["annotations"] if a.get("iscrowd", 0) == 0]
    removed = before - len(data["annotations"])
    if removed:
        import logging
        logging.getLogger(__name__).info("filter_crowd: removed %d crowd annotations", removed)
    if out_path is not None:
        Path(out_path).write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------------------
# #4 + #5  pycocotools evaluation (standard AP@100, optional explicitly requested cap)
# ---------------------------------------------------------------------------

def coco_evaluate(
    annotations_json: str | Path,
    predictions: list[dict],
    max_dets: list[int] | None = None,
    image_ids: list[int] | None = None,
) -> dict:
    """Evaluate detection predictions with pycocotools.

    Uses the official evaluator so iscrowd / ignore regions are handled
    correctly (custom mAP implementations commonly skip this, inflating FP
    counts). Defaults to standard COCO AP@100. With a custom cap, AP is
    summarized explicitly from that slice; the returned maxDets identifies it.

    Args:
        annotations_json: Path to a COCO instances JSON file.
        predictions:      List of dicts with keys image_id, category_id,
                          bbox ([x, y, w, h] pixels), score.
        max_dets:         Per-image detection cap at each level.
                          Defaults to [1, 10, 100]; must be ascending.
        image_ids:        Exact evaluated image subset. Defaults to all GT images.

    Returns:
        dict with AP, AP50, AP75, APs, APm, APl (COCO primary metrics).
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("pip install pycocotools") from exc

    if max_dets is None:
        max_dets = [1, 10, 100]
    if (len(max_dets) != 3 or any(not isinstance(value, int) or value <= 0 for value in max_dets)
            or max_dets != sorted(max_dets)):
        raise ValueError("max_dets must contain three positive ascending integers")

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotations_json))
        if predictions:
            coco_dt = coco_gt.loadRes(predictions)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "images": list(coco_gt.dataset.get("images", [])),
                "categories": list(coco_gt.dataset.get("categories", [])),
                "annotations": [],
            }
            coco_dt.createIndex()
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = max_dets
        if image_ids is not None:
            ev.params.imgIds = sorted(set(int(i) for i in image_ids))
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # COCO's stats[0] hard-codes maxDets=100, even when params.maxDets differs.
    # Read precision directly for a consistent custom-cap summary.
    precision = ev.eval["precision"]  # IoU, recall, category, area, maxDets
    cap_index = list(ev.params.maxDets).index(max_dets[-1])

    def ap(area="all", threshold=None):
        values = precision[:, :, :, ev.params.areaRngLbl.index(area), cap_index]
        if threshold is not None:
            values = values[abs(ev.params.iouThrs - threshold) < 1e-6]
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else -1.0

    return {"AP": ap(), "AP50": ap(threshold=0.5), "AP75": ap(threshold=0.75),
            "APs": ap("small"), "APm": ap("medium"), "APl": ap("large"),
            "maxDets": max_dets[-1]}


# ---------------------------------------------------------------------------
# #7  Focal loss
# ---------------------------------------------------------------------------

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss for multi-label classification heads.

    Reduces gradient contribution from easy negatives (large pt), focusing
    training on the ~10-20% of instances that COCO leaves unlabeled and that
    the model currently treats as hard false positives.

    Args:
        logits:  Raw (pre-sigmoid) predictions, any shape.
        targets: Binary targets, same shape as logits.
        alpha:   Foreground weighting factor.
        gamma:   Focusing exponent; 0 recovers standard BCE.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1.0 - pt) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# #10  Label smoothing
# ---------------------------------------------------------------------------

def smooth_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 0.1,
) -> torch.Tensor:
    """BCE with label smoothing.

    Pulls positives down to (1 - eps/2) and pushes negatives up to eps/2,
    reducing overconfidence at noisy class boundaries (cup/bowl, truck/car,
    handbag/backpack, etc.).  Recommended eps=0.05-0.1.
    """
    targets = targets.clamp(eps / 2.0, 1.0 - eps / 2.0)
    return F.binary_cross_entropy_with_logits(logits, targets)


# ---------------------------------------------------------------------------
# #8  Repeat factor sampler + inverse-frequency class weights
# ---------------------------------------------------------------------------

def repeat_factors(
    category_ids_per_image: list[list[int]],
    t: float = 0.001,
) -> list[float]:
    """Per-image repeat factors for oversampling rare-category images (LVIS-style).

    Each image's factor = max over its categories of max(1, sqrt(t / freq_c)).
    Images whose rarest category appears with frequency < t get factor > 1 and
    are sampled multiple times per epoch.

    Args:
        category_ids_per_image: Category IDs present in each training image.
        t:                      Frequency threshold (default 0.001 from LVIS paper).

    Returns:
        List of floats >= 1.0, one per image.
    """
    all_ids = [c for cats in category_ids_per_image for c in cats]
    total = len(all_ids)
    freq = {k: v / total for k, v in Counter(all_ids).items()}

    factors = []
    for cats in category_ids_per_image:
        if not cats:
            factors.append(1.0)
        else:
            factors.append(max(max(1.0, math.sqrt(t / freq[c])) for c in cats))
    return factors


def class_weights(category_ids_per_image: list[list[int]]) -> dict[int, float]:
    """Inverse-frequency weights for the classification loss.

    Weight_c = N_total / (n_classes * count_c), so every class contributes
    equally to the total loss regardless of instance count.
    """
    all_ids = [c for cats in category_ids_per_image for c in cats]
    counts = Counter(all_ids)
    n_classes = len(counts)
    total = sum(counts.values())
    return {k: total / (n_classes * v) for k, v in counts.items()}


class RepeatFactorSampler(Sampler):
    """Weighted sampler driven by per-image repeat factors.

    Each epoch draws ceil(sum(factors)) indices so rare-category images appear
    multiple times without removing any image entirely.

    Usage::

        factors = repeat_factors(cat_ids_per_image, t=0.001)
        sampler = RepeatFactorSampler(factors)
        loader = DataLoader(dataset, sampler=sampler, batch_size=16)
    """

    def __init__(self, factors: list[float], seed: int = 0) -> None:
        self._factors = factors
        self._seed = seed
        self._n = math.ceil(sum(factors))

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        weights = torch.tensor(self._factors, dtype=torch.float64)
        yield from torch.multinomial(weights, self._n, replacement=True, generator=g).tolist()


# ---------------------------------------------------------------------------
# #11  Multi-scale training collate
# ---------------------------------------------------------------------------

_DEFAULT_SCALES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]


class MultiScaleCollate:
    """Collate function that randomly resizes each batch to a new square size.

    Resamples every *interval* batches to keep GPU memory predictable.
    All sizes should be multiples of the model stride (typically 32).

    Usage::

        collate = MultiScaleCollate()
        loader = DataLoader(dataset, collate_fn=collate, batch_size=16)

    The returned batch dict matches the format expected by DetectionLoss:
    ``images`` (B, 3, H, W), ``bboxes`` (N, 4) pixel xyxy, ``cls`` (N,),
    ``batch_idx`` (N,) mapping each box to its image in the batch.
    """

    def __init__(
        self,
        sizes: list[int] = _DEFAULT_SCALES,
        interval: int = 10,
    ) -> None:
        self.sizes = sizes
        self.interval = interval
        self._count = 0
        self._size = sizes[len(sizes) // 2]

    def __call__(self, batch: list[dict]) -> dict:
        if self._count % self.interval == 0:
            self._size = random.choice(self.sizes)
        self._count += 1
        size = self._size

        imgs, bboxes, clss, idxs = [], [], [], []
        for i, sample in enumerate(batch):
            img = torch.as_tensor(sample["image"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            h, w = img.shape[1], img.shape[2]
            img = F.interpolate(img.unsqueeze(0), size=(size, size),
                                mode="bilinear", align_corners=False).squeeze(0)
            boxes = torch.as_tensor(sample["boxes"], dtype=torch.float32)
            if boxes.shape[0]:
                boxes[:, [0, 2]] *= size / w
                boxes[:, [1, 3]] *= size / h
            imgs.append(img)
            bboxes.append(boxes)
            clss.append(torch.as_tensor(sample["cls"], dtype=torch.float32))
            idxs.append(torch.full((len(sample["cls"]),), i, dtype=torch.long))

        return {
            "images": torch.stack(imgs),
            "bboxes": torch.cat(bboxes) if bboxes else torch.zeros((0, 4)),
            "cls": torch.cat(clss) if clss else torch.zeros(0),
            "batch_idx": torch.cat(idxs) if idxs else torch.zeros(0, dtype=torch.long),
        }
