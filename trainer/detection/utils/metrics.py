"""COCO-style detection metrics: AP per class, mAP50, and mAP50-95.

Implements 101-point interpolated AP over IoU thresholds 0.50:0.05:0.95,
confidence-ordered greedy matching of predictions to ground truth per class,
plus a confusion matrix. For official COCO numbers use ``coco_evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from trainer.detection.utils.boxes import box_iou

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
EVALUATOR_VERSION = "trainer-greedy-v2"


def greedy_matches(iou: np.ndarray, eligible: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Match confidence-ordered columns to their best unmatched eligible GT row."""
    available = np.ones(iou.shape[0], dtype=bool)
    pairs = []
    for p in np.flatnonzero((eligible & (iou >= threshold)).any(axis=0)):
        candidates = np.flatnonzero(available & eligible[:, p] & (iou[:, p] >= threshold))
        if candidates.size:
            g = int(candidates[np.argmax(iou[candidates, p])])
            available[g] = False
            pairs.append((g, p))
    return pairs


def match_predictions(pred_boxes: torch.Tensor, pred_cls: torch.Tensor, gt_boxes: torch.Tensor,
                      gt_cls: torch.Tensor) -> np.ndarray:
    """True-positive matrix for one image.

    Returns a bool array (n_pred, n_iou_thresholds): pred i is a TP at
    threshold t. Predictions must be ordered by descending confidence. Each
    ordinary GT can be used once; a prediction tries its best unmatched GT.
    """
    n_pred = pred_boxes.shape[0]
    tp = np.zeros((n_pred, len(IOU_THRESHOLDS)), dtype=bool)
    if n_pred == 0 or gt_boxes.shape[0] == 0:
        return tp
    iou = box_iou(gt_boxes, pred_boxes).cpu().numpy()  # (n_gt, n_pred)
    cls_match = (gt_cls.view(-1, 1) == pred_cls.view(1, -1)).cpu().numpy()

    for ti, thr in enumerate(IOU_THRESHOLDS):
        for _, p in greedy_matches(iou, cls_match, thr):
            tp[p, ti] = True
    return tp


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP from raw recall/precision curves."""
    if recall.size == 0:
        return 0.0
    mpre = np.maximum.accumulate(precision[::-1])[::-1]
    recall_points = np.linspace(0.0, 1.0, 101)
    indices = np.searchsorted(recall, recall_points, side="left")
    sampled = np.zeros(101)
    valid = indices < len(mpre)
    sampled[valid] = mpre[indices[valid]]
    return float(sampled.mean())


@dataclass
class DetMetrics:
    """Accumulates matches across a dataset and computes final metrics."""

    nc: int
    names: dict[int, str] = field(default_factory=dict)
    _tp: list[np.ndarray] = field(default_factory=list)
    _conf: list[np.ndarray] = field(default_factory=list)
    _pred_cls: list[np.ndarray] = field(default_factory=list)
    _gt_cls: list[np.ndarray] = field(default_factory=list)

    def update(self, pred: torch.Tensor, gt_boxes: torch.Tensor, gt_cls: torch.Tensor) -> None:
        """pred: (n, 6) [x1, y1, x2, y2, conf, cls], sorted or unsorted."""
        if pred.numel():
            order = pred[:, 4].argsort(descending=True, stable=True)
            pred = pred[order]
        tp = match_predictions(pred[:, :4], pred[:, 5], gt_boxes, gt_cls)
        self._tp.append(tp)
        self._conf.append(pred[:, 4].cpu().numpy())
        self._pred_cls.append(pred[:, 5].cpu().numpy())
        self._gt_cls.append(gt_cls.cpu().numpy())

    def compute(self, curves: bool = False) -> dict:
        if not self._tp:
            return self._empty(curves=curves)
        tp = np.concatenate(self._tp, 0)
        conf = np.concatenate(self._conf, 0)
        pred_cls = np.concatenate(self._pred_cls, 0)
        gt_cls = np.concatenate(self._gt_cls, 0)

        order = np.argsort(-conf, kind="stable")
        tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]

        classes = np.unique(gt_cls).astype(int)
        ap = np.zeros((self.nc, len(IOU_THRESHOLDS)))
        p50 = np.zeros(self.nc)
        r50 = np.zeros(self.nc)
        eps = 1e-9
        for c in classes:
            mask = pred_cls == c
            n_gt = int((gt_cls == c).sum())
            n_p = int(mask.sum())
            if n_p == 0 or n_gt == 0:
                continue
            tpc = tp[mask].cumsum(0)
            fpc = (~tp[mask]).cumsum(0)
            recall = tpc / n_gt
            precision = tpc / np.maximum(tpc + fpc, 1)
            for ti in range(len(IOU_THRESHOLDS)):
                ap[c, ti] = compute_ap(recall[:, ti], precision[:, ti])
            # P/R at max-F1 operating point on the IoU=0.5 curve
            f1 = 2 * precision[:, 0] * recall[:, 0] / (precision[:, 0] + recall[:, 0] + eps)
            best = int(f1.argmax())
            p50[c] = precision[best, 0]
            r50[c] = recall[best, 0]

        seen = np.zeros(self.nc, dtype=bool)
        seen[classes] = True
        n_seen = max(int(seen.sum()), 1)
        ap50_per_class = ap[:, 0]
        ap_per_class = ap.mean(1)

        result = {
            "mAP50": float(ap50_per_class[seen].sum() / n_seen),
            "mAP50-95": float(ap_per_class[seen].sum() / n_seen),
            "precision": float(p50[seen].sum() / n_seen),
            "recall": float(r50[seen].sum() / n_seen),
            "ap50_per_class": ap50_per_class,
            "ap_per_class": ap_per_class,
            "precision_per_class": p50,
            "recall_per_class": r50,
            "classes_seen": seen,
        }

        if curves:
            recall_pts = np.linspace(0, 1, 101)
            conf_pts = np.linspace(0, 1, 201)
            mean_pr = np.zeros(101)
            mean_f1 = np.zeros(201)
            n_curve = 0
            for c in classes:
                mask = pred_cls == c
                n_gt = int((gt_cls == c).sum())
                if not mask.any() or n_gt == 0:
                    continue
                tpc = tp[mask].cumsum(0)
                fpc = (~tp[mask]).cumsum(0)
                r0 = tpc[:, 0] / n_gt
                p0 = tpc[:, 0] / np.maximum(tpc[:, 0] + fpc[:, 0], 1)
                f1_0 = 2 * p0 * r0 / (p0 + r0 + eps)
                c_c = conf[mask]  # descending
                mean_pr += np.interp(recall_pts,
                                     np.concatenate([[0.0], r0, [1.0]]),
                                     np.concatenate([[1.0], p0, [0.0]]))
                c_asc = np.concatenate([[0.0], c_c[::-1], [min(float(c_c[0]) + 1e-6, 1.0)]])
                f1_asc = np.concatenate([[float(f1_0[-1])], f1_0[::-1], [0.0]])
                mean_f1 += np.interp(conf_pts, c_asc, f1_asc)
                n_curve += 1
            if n_curve:
                mean_pr /= n_curve
                mean_f1 /= n_curve
            result.update({
                "pr_recall": recall_pts,
                "pr_precision": mean_pr,
                "f1_conf": conf_pts,
                "f1_mean": mean_f1,
            })

        return result

    def _empty(self, curves: bool = False) -> dict:
        result = {
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "ap50_per_class": np.zeros(self.nc),
            "ap_per_class": np.zeros(self.nc),
            "precision_per_class": np.zeros(self.nc),
            "recall_per_class": np.zeros(self.nc),
            "classes_seen": np.zeros(self.nc, dtype=bool),
        }
        if curves:
            result.update({
                "pr_recall": np.linspace(0, 1, 101),
                "pr_precision": np.zeros(101),
                "f1_conf": np.linspace(0, 1, 201),
                "f1_mean": np.zeros(201),
            })
        return result


class ConfusionMatrix:
    """Detection confusion matrix at fixed conf/IoU (background = last index)."""

    def __init__(self, nc: int, conf: float = 0.25, iou_thres: float = 0.45):
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        self.matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    def update(self, pred: torch.Tensor, gt_boxes: torch.Tensor, gt_cls: torch.Tensor) -> None:
        pred = pred[pred[:, 4] > self.conf]
        pred = pred[pred[:, 4].argsort(descending=True, stable=True)]
        gt_cls_np = gt_cls.int().cpu().numpy()
        if pred.shape[0] == 0:
            for gc in gt_cls_np:
                self.matrix[self.nc, gc] += 1  # missed gt
            return
        pred_cls = pred[:, 5].int().cpu().numpy()
        if gt_boxes.shape[0] == 0:
            for pc in pred_cls:
                self.matrix[pc, self.nc] += 1  # false positive
            return
        iou = box_iou(gt_boxes, pred[:, :4]).cpu().numpy()
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for g, p in greedy_matches(iou, np.ones_like(iou, dtype=bool), self.iou_thres):
            matched_gt.add(g)
            matched_pred.add(p)
            self.matrix[pred_cls[p], gt_cls_np[g]] += 1
        for g in range(len(gt_cls_np)):
            if g not in matched_gt:
                self.matrix[self.nc, gt_cls_np[g]] += 1
        for p in range(len(pred_cls)):
            if p not in matched_pred:
                self.matrix[pred_cls[p], self.nc] += 1


@dataclass
class PerImageStats:
    """Per-image, per-class TP/FP/FN/TN counts and FP confusion at IoU 0.5.

    Matching is performed without a class constraint so cross-class box overlaps
    are recorded as FP/FN pairs rather than being silently ignored.
    """

    nc: int
    names: dict[int, str] = field(default_factory=dict)
    iou_thres: float = 0.5
    _records: list[dict] = field(default_factory=list)

    def update(self, image_id: str, pred: torch.Tensor, gt_boxes: torch.Tensor,
               gt_cls: torch.Tensor) -> None:
        gt_cls_np = gt_cls.int().cpu().numpy()
        n_gt = len(gt_cls_np)

        tp_c = np.zeros(self.nc, dtype=np.int64)
        fp_c = np.zeros(self.nc, dtype=np.int64)
        fn_c = np.zeros(self.nc, dtype=np.int64)
        fp_details: list[dict] = []

        if pred.numel() > 0:
            order = pred[:, 4].argsort(descending=True, stable=True)
            pred_sorted = pred[order].cpu().numpy()
        else:
            pred_sorted = np.zeros((0, 6), dtype=np.float32)

        n_pred = len(pred_sorted)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        if n_pred > 0 and n_gt > 0:
            pred_boxes_t = torch.from_numpy(pred_sorted[:, :4].astype(np.float32))
            if gt_boxes.is_cuda:
                pred_boxes_t = pred_boxes_t.to(gt_boxes.device)
            iou = box_iou(gt_boxes, pred_boxes_t).cpu().numpy()  # (n_gt, n_pred)
            for g, p in greedy_matches(iou, np.ones_like(iou, dtype=bool), self.iou_thres):
                matched_gt.add(g)
                matched_pred.add(p)
                pc = int(pred_sorted[p, 5])
                gc = int(gt_cls_np[g])
                if pc == gc:
                    tp_c[pc] += 1
                else:
                    fp_c[pc] += 1
                    fn_c[gc] += 1
                    fp_details.append({"pred_cls": pc, "gt_cls": gc})

        for p in range(n_pred):
            if p not in matched_pred:
                pc = int(pred_sorted[p, 5])
                fp_c[pc] += 1
                fp_details.append({"pred_cls": pc, "gt_cls": None})

        for g in range(n_gt):
            if g not in matched_gt:
                fn_c[int(gt_cls_np[g])] += 1

        gt_cls_set = {int(c) for c in gt_cls_np}
        pred_cls_set = {int(pred_sorted[p, 5]) for p in range(n_pred)}
        tn_c = np.array(
            [1 if (c not in gt_cls_set and c not in pred_cls_set) else 0
             for c in range(self.nc)],
            dtype=np.int64,
        )

        self._records.append({
            "image_id": image_id,
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "tn": tn_c,
            "fp_details": fp_details,
        })

    def per_image_class_rows(self) -> list[list]:
        """Rows for the per-image/per-class TP/FP/FN/TN table.

        Rows where TP=FP=FN=0 (class correctly absent) are omitted to keep the
        table readable; those images contribute to TN for the aggregate view.
        """
        rows = []
        for rec in self._records:
            img = rec["image_id"].rsplit("/", 1)[-1]
            for c in range(self.nc):
                tp = int(rec["tp"][c])
                fp = int(rec["fp"][c])
                fn = int(rec["fn"][c])
                tn = int(rec["tn"][c])
                if tp == 0 and fp == 0 and fn == 0:
                    continue
                name = self.names.get(c, str(c))
                confused = [
                    self.names.get(d["gt_cls"], str(d["gt_cls"]))
                    if d["gt_cls"] is not None else "background"
                    for d in rec["fp_details"] if d["pred_cls"] == c
                ]
                rows.append([img, name, tp, fp, fn, tn,
                              ", ".join(confused) if confused else "-"])
        return rows

    def fp_detail_rows(self) -> list[list]:
        """One row per FP event: image, predicted class, ground-truth class."""
        rows = []
        for rec in self._records:
            if not rec["fp_details"]:
                continue
            img = rec["image_id"].rsplit("/", 1)[-1]
            for d in rec["fp_details"]:
                pred_name = self.names.get(d["pred_cls"], str(d["pred_cls"]))
                gt_name = (
                    self.names.get(d["gt_cls"], str(d["gt_cls"]))
                    if d["gt_cls"] is not None else "background"
                )
                rows.append([img, pred_name, gt_name])
        return rows
