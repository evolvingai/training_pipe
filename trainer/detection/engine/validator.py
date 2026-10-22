"""Validation: run the model over a dataloader and compute in-house mAP."""

from __future__ import annotations

from pathlib import Path

import torch

from trainer.detection.utils.boxes import xywh2xyxy
from trainer.detection.utils.logging import LOGGER
from trainer.detection.utils.metrics import ConfusionMatrix, DetMetrics, PerImageStats
from trainer.detection.utils.nms import non_max_suppression

_PLOT_BATCHES = 2  # number of batches to visualize when plot_dir is set


def _save_batch_plots(batch: dict, preds: list, names: dict[int, str],
                      plot_dir: Path, batch_num: int, scale_cpu: torch.Tensor) -> None:
    import cv2
    from trainer.detection.utils.plotting import draw_boxes, make_grid

    plot_dir.mkdir(parents=True, exist_ok=True)
    label_imgs, pred_imgs = [], []
    for i in range(len(batch["paths"])):
        img = batch["img"][i].permute(1, 2, 0).numpy().copy()  # uint8 HWC BGR
        sel = batch["batch_idx"] == i
        gt_boxes = xywh2xyxy(batch["bboxes"][sel] * scale_cpu).numpy()
        gt_cls = batch["cls"][sel].numpy()
        label_imgs.append(draw_boxes(img, gt_boxes, gt_cls, names))
        p = preds[i].cpu()
        if p.shape[0]:
            pred_imgs.append(draw_boxes(img, p[:, :4].numpy(), p[:, 5].numpy(), names,
                                        confs=p[:, 4].numpy()))
        else:
            pred_imgs.append(img.copy())
    cv2.imwrite(str(plot_dir / f"val_batch{batch_num}_labels.jpg"), make_grid(label_imgs))
    cv2.imwrite(str(plot_dir / f"val_batch{batch_num}_pred.jpg"), make_grid(pred_imgs))


@torch.no_grad()
def validate(model, dataloader, cfg: dict, names: dict[int, str], device=None,
             compute_confusion: bool = False, compute_per_image: bool = False,
             half: bool = False, plot_dir: Path | None = None,
             compute_curves: bool = False) -> dict:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    nc = len(names)
    metrics = DetMetrics(nc=nc, names=names)
    confusion = ConfusionMatrix(nc) if compute_confusion else None
    per_image = PerImageStats(nc=nc, names=names) if compute_per_image else None
    val_loss_total = 0.0
    n_batches = 0
    total_batches = len(dataloader)

    for done, batch in enumerate(dataloader, 1):
        imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
        if half:
            imgs = imgs.half()
        decoded, _ = model(imgs)
        preds = non_max_suppression(decoded.float(), conf_thres=float(cfg["val_conf"]),
                                    iou_thres=float(cfg["iou"]), max_det=int(cfg["max_det"]))
        h, w = imgs.shape[-2:]
        scale_vec = torch.tensor([w, h, w, h], device=device)
        for i, pred in enumerate(preds):
            sel = batch["batch_idx"] == i
            gt_boxes = xywh2xyxy(batch["bboxes"][sel].to(device) * scale_vec)
            gt_cls = batch["cls"][sel].to(device)
            metrics.update(pred, gt_boxes, gt_cls)
            if confusion is not None:
                confusion.update(pred, gt_boxes, gt_cls)
            if per_image is not None:
                per_image.update(batch["paths"][i], pred, gt_boxes, gt_cls)

        if plot_dir is not None and (done - 1) < _PLOT_BATCHES:
            _save_batch_plots(batch, preds, names, Path(plot_dir), done - 1,
                              scale_vec.cpu())
        n_batches += 1
        if done % max(1, total_batches // 10) == 0 or done == total_batches:
            filled = int(25 * done / total_batches)
            bar = '█' * filled + '░' * (25 - filled)
            LOGGER.info(f"  [{bar}] {done}/{total_batches} batches, {total_batches - done} remaining")

    result = metrics.compute(curves=compute_curves)
    result["val_loss"] = val_loss_total / max(n_batches, 1)
    if confusion is not None:
        result["confusion"] = confusion.matrix
    if per_image is not None:
        result["per_image_stats"] = per_image
    if was_training:
        model.train()
    return result


def per_class_table(result: dict, names: dict[int, str]) -> str:
    """Fixed-width per-class metrics table for console output."""
    lines = [f"{'class':<20s} {'P':>8s} {'R':>8s} {'AP50':>8s} {'AP50-95':>8s}"]
    ap50 = result["ap50_per_class"]
    ap = result["ap_per_class"]
    p = result["precision_per_class"]
    r = result["recall_per_class"]
    seen = result["classes_seen"]
    for c, name in names.items():
        if seen[c]:
            lines.append(f"{name:<20s} {p[c]:>8.3f} {r[c]:>8.3f} {ap50[c]:>8.3f} {ap[c]:>8.3f}")
    lines.append(f"{'all':<20s} {result['precision']:>8.3f} {result['recall']:>8.3f} "
                 f"{result['mAP50']:>8.3f} {result['mAP50-95']:>8.3f}")
    return "\n".join(lines)


def log_per_class_metrics(tracker, result: dict, names: dict[int, str], step: int) -> None:
    """Per-class metrics to all tracker backends (sanitized metric keys)."""
    payload: dict[str, float] = {}
    for c, name in names.items():
        key = name.replace(" ", "_")
        payload[f"val/P_{key}"] = float(result["precision_per_class"][c])
        payload[f"val/R_{key}"] = float(result["recall_per_class"][c])
        payload[f"val/AP50_{key}"] = float(result["ap50_per_class"][c])
        payload[f"val/AP50-95_{key}"] = float(result["ap_per_class"][c])
    tracker.log_metrics(payload, step=step)


def print_val_summary(result: dict, names: dict[int, str]) -> None:
    LOGGER.info(per_class_table(result, names))
