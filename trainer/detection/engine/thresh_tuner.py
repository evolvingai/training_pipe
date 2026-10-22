"""Per-class confidence threshold tuning.

Sweeps the confidence axis per class on validation predictions, picks the
threshold maximizing a fitness function (f1 / precision-floor / recall-floor),
and flags poorly separated classes whose fitness never reaches a minimum.
"""

from __future__ import annotations

import numpy as np
import torch

from trainer.detection.config import dump as toml_dump
from trainer.detection.utils.boxes import xywh2xyxy
from trainer.detection.utils.logging import LOGGER
from trainer.detection.utils.metrics import match_predictions
from trainer.detection.utils.nms import non_max_suppression


def collect_stats(model, dataloader, cfg: dict, nc: int, device=None) -> list[dict]:
    """Low-threshold predictions matched against gt (IoU 0.5), per image."""
    device = device or next(model.parameters()).device
    model.eval()
    stats = []
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["img"].to(device).float() / 255.0
            decoded, _ = model(imgs)
            preds = non_max_suppression(decoded.float(), conf_thres=0.001,
                                        iou_thres=float(cfg["iou"]), max_det=int(cfg["max_det"]))
            h, w = imgs.shape[-2:]
            scale_vec = torch.tensor([w, h, w, h], device=device)
            for i, pred in enumerate(preds):
                sel = batch["batch_idx"] == i
                gt_boxes = xywh2xyxy(batch["bboxes"][sel].to(device) * scale_vec)
                gt_cls = batch["cls"][sel].to(device)
                tp = match_predictions(pred[:, :4], pred[:, 5], gt_boxes, gt_cls)[:, 0]  # IoU=0.5
                stats.append({
                    "conf": pred[:, 4].cpu().numpy(),
                    "cls": pred[:, 5].cpu().numpy().astype(int),
                    "tp": tp,
                    "gt_cls": gt_cls.cpu().numpy().astype(int),
                })
    return stats


def _fitness(precision: np.ndarray, recall: np.ndarray, kind: str) -> np.ndarray:
    eps = 1e-9
    if kind == "f1":
        return 2 * precision * recall / (precision + recall + eps)
    if kind.startswith("precision>="):  # maximize recall subject to precision floor
        floor = float(kind.split(">=")[1])
        return np.where(precision >= floor, recall, -1.0)
    if kind.startswith("recall>="):  # maximize precision subject to recall floor
        floor = float(kind.split(">=")[1])
        return np.where(recall >= floor, precision, -1.0)
    raise ValueError(f"unknown thresh fitness '{kind}'")


def tune_thresholds(stats: list[dict], nc: int, names: dict[int, str], cfg: dict) -> dict:
    """Returns {class_name: threshold} plus a diagnostics dict."""
    kind = str(cfg["thresh_fitness"])
    steps = int(cfg["thresh_sweep_steps"])
    min_fit = float(cfg["thresh_min_fitness"])
    default = float(cfg["conf"])

    conf = np.concatenate([s["conf"] for s in stats]) if stats else np.zeros(0)
    cls = np.concatenate([s["cls"] for s in stats]) if stats else np.zeros(0, int)
    tp = np.concatenate([s["tp"] for s in stats]) if stats else np.zeros(0, bool)
    gt_cls = np.concatenate([s["gt_cls"] for s in stats]) if stats else np.zeros(0, int)

    grid = np.linspace(0.01, 0.99, steps)
    thresholds: dict[str, float] = {}
    diagnostics: dict[str, dict] = {}
    for c in range(nc):
        name = names.get(c, str(c))
        n_gt = int((gt_cls == c).sum())
        mask = cls == c
        if n_gt == 0 or mask.sum() == 0:
            thresholds[name] = default
            diagnostics[name] = {"status": "no_data", "best_fitness": 0.0, "threshold": default}
            continue
        c_conf, c_tp = conf[mask], tp[mask]
        # vectorized sweep: at threshold t, TP = tp[conf>=t].sum(), P = TP/n_kept, R = TP/n_gt
        keep = c_conf[None, :] >= grid[:, None]  # (steps, n)
        tps = (keep & c_tp[None, :]).sum(1).astype(np.float64)
        kept = keep.sum(1).astype(np.float64)
        precision = np.where(kept > 0, tps / np.maximum(kept, 1), 0.0)
        recall = tps / n_gt
        fit = _fitness(precision, recall, kind)
        best = int(fit.argmax())
        best_fit = float(fit[best])
        thr = float(grid[best]) if best_fit >= 0 else default
        thresholds[name] = round(thr, 4)
        status = "ok" if best_fit >= min_fit else "poorly_separated"
        diagnostics[name] = {
            "status": status,
            "best_fitness": round(best_fit, 4),
            "threshold": thresholds[name],
            "precision": round(float(precision[best]), 4),
            "recall": round(float(recall[best]), 4),
            "n_gt": n_gt,
        }
        if status != "ok":
            LOGGER.warning(f"class '{name}' is poorly separated: best {kind}="
                           f"{best_fit:.3f} < {min_fit} (needs more/better data)")
    return {"thresholds": thresholds, "diagnostics": diagnostics}


def save_thresholds(result: dict, path) -> None:
    toml_dump({"thresholds": result["thresholds"]}, path)
    LOGGER.info(f"saved per-class thresholds to {path}")
