"""Validation: run a model over a dataloader and compute detection metrics."""

from __future__ import annotations

from pathlib import Path
from functools import wraps

import torch

from trainer.detection.utils.boxes import scale_boxes, xywh2xyxy
from trainer.detection.nn.loss import DetectionLoss
from trainer.detection.utils.logging import LOGGER
from trainer.detection.utils.metrics import EVALUATOR_VERSION, ConfusionMatrix, DetMetrics, PerImageStats
from trainer.detection.utils.nms import non_max_suppression

_PLOT_BATCHES = 2  # number of batches to visualize when plot_dir is set


def preserve_training_mode(fn):
    @wraps(fn)
    def wrapped(model, *args, **kwargs):
        modes = [(module, module.training) for module in model.modules()]
        try:
            return fn(model, *args, **kwargs)
        finally:
            for module, training in modes:
                module.training = training
    return wrapped


def _save_batch_plots(batch: dict, preds: list, names: dict[int, str],
                      plot_dir: Path, batch_num: int, scale_cpu: torch.Tensor) -> None:
    from trainer.detection.utils.image import imwrite_rgb
    from trainer.detection.utils.plotting import draw_boxes, make_grid

    plot_dir.mkdir(parents=True, exist_ok=True)
    label_imgs, pred_imgs = [], []
    for i in range(len(batch["paths"])):
        img = batch["img"][i].permute(1, 2, 0).numpy().copy()  # uint8 HWC RGB
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
    imwrite_rgb(plot_dir / f"val_batch{batch_num}_labels.jpg", make_grid(label_imgs))
    imwrite_rgb(plot_dir / f"val_batch{batch_num}_pred.jpg", make_grid(pred_imgs))


@preserve_training_mode
@torch.no_grad()
def validate(model, dataloader, cfg: dict, names: dict[int, str], device=None,
             compute_confusion: bool = False, compute_per_image: bool = False,
             half: bool = False, plot_dir: Path | None = None,
             compute_curves: bool = False, compute_loss: bool = False,
             prefetched_iter=None) -> dict:
    device = device or next(model.parameters()).device
    nc = len(names)
    if int(getattr(model, "nc", nc)) != nc:
        raise ValueError(f"model has {getattr(model, 'nc', '?')} classes but names defines {nc}")
    was_training = model.training
    model.eval()
    metrics = DetMetrics(nc=nc, names=names)
    confusion = (ConfusionMatrix(
        nc, conf=float(cfg["confusion_conf"]), iou_thres=float(cfg["confusion_iou"]),
    ) if compute_confusion else None)
    per_image = PerImageStats(nc=nc, names=names) if compute_per_image else None
    loss_fn = DetectionLoss(model, cfg) if compute_loss else None
    val_loss_total = 0.0
    n_images = 0
    total_batches = len(dataloader)
    compare_pycoco = bool(cfg.get("compare_pycoco", False))
    coco_predictions: list[dict] = []
    coco_image_ids: list[int] = []
    dataset = dataloader.dataset
    if compare_pycoco and not all(hasattr(dataset, key) for key in (
        "annotations_json", "image_id_by_path", "category_id_for_class",
    )):
        raise ValueError("compare_pycoco=true requires a COCO-format validation split")

    for done, batch in enumerate(prefetched_iter if prefetched_iter is not None else dataloader, 1):
        imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
        with torch.autocast(device_type=device.type, enabled=half):
            decoded, feats = model(imgs)
            if loss_fn is not None:
                batch_loss, _ = loss_fn(feats, batch)
                val_loss_total += float(batch_loss)
        n_images += int(imgs.shape[0])
        preds = non_max_suppression(decoded.float(), conf_thres=float(cfg["val_conf"]),
                                    iou_thres=float(cfg["iou"]), max_det=int(cfg["max_det"]),
                                    multi_label=bool(cfg.get("nms_multi_label", False)))
        h, w = imgs.shape[-2:]
        scale_vec = torch.tensor([w, h, w, h], device=device)
        batch_idx = batch["batch_idx"].to(device)
        bboxes = batch["bboxes"].to(device)
        cls = batch["cls"].to(device)
        for i, pred in enumerate(preds):
            sel = batch_idx == i
            gt_boxes = xywh2xyxy(bboxes[sel] * scale_vec)
            gt_cls = cls[sel]
            metrics.update(pred, gt_boxes, gt_cls)
            if confusion is not None:
                confusion.update(pred, gt_boxes, gt_cls)
            if per_image is not None:
                per_image.update(batch["paths"][i], pred, gt_boxes, gt_cls)
            if compare_pycoco:
                path = str(batch["paths"][i])
                image_id = dataset.image_id_by_path.get(path, dataset.image_id_by_path.get(str(Path(path).resolve())))
                if image_id is None:
                    raise KeyError(f"COCO image id not found for validation image: {path}")
                coco_image_ids.append(int(image_id))
                if pred.shape[0]:
                    original = scale_boxes(
                        pred[:, :4], (h, w), batch["orig_shapes"][i], pad=batch["pads"][i],
                        gain=float(batch["ratios"][i]),
                    )
                    boxes_xywh = original.cpu()
                    boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
                    for box, score, pred_cls in zip(boxes_xywh, pred[:, 4].cpu(), pred[:, 5].cpu()):
                        coco_predictions.append({
                            "image_id": int(image_id),
                            "category_id": int(dataset.category_id_for_class[int(pred_cls.item())]),
                            "bbox": [float(v) for v in box.tolist()],
                            "score": float(score.item()),
                        })

        if plot_dir is not None and (done - 1) < _PLOT_BATCHES:
            _save_batch_plots(batch, preds, names, Path(plot_dir), done - 1,
                              scale_vec.cpu())
        if done % 5 == 0 or done == total_batches:
            filled = int(25 * done / total_batches)
            bar = '█' * filled + '░' * (25 - filled)
            print(f"\r  validating  [{bar}] {done}/{total_batches} batches, {total_batches - done} remaining",
                  end="", flush=True)
    print()

    result = metrics.compute(curves=compute_curves)
    result["val_loss"] = val_loss_total / max(n_images, 1) if compute_loss else None
    result["evaluation"] = {
        "version": EVALUATOR_VERSION, "primary": "trainer",
        "max_det": int(cfg["max_det"]), "val_conf": float(cfg["val_conf"]),
        "nms_iou": float(cfg["iou"]), "nms_multi_label": bool(cfg.get("nms_multi_label", False)),
        "precision_recall": "per-class best F1 at IoU 0.5",
        "crowd_policy": "crowd GT omitted; no crowd-region detection suppression",
        "confusion_matching": "class-agnostic confidence-greedy",
    }
    if confusion is not None:
        result["confusion"] = confusion.matrix
    if per_image is not None:
        result["per_image_stats"] = per_image
    if compare_pycoco:
        from trainer.detection.data.coco_sota_fixes import coco_evaluate

        official = coco_evaluate(
            dataset.annotations_json, coco_predictions,
            max_dets=[1, 10, 100],
            image_ids=coco_image_ids,
        )
        result["pycoco"] = official
        result["evaluation"]["pycoco_max_dets"] = [1, 10, 100]
        result["evaluation"]["comparison_note"] = (
            "Same post-NMS predictions; caps and crowd/ignore semantics may differ. "
            "Predictions remain limited by inference max_det."
        )
        result["pycoco_delta"] = {
            "AP50": float(result["mAP50"] - official["AP50"]),
            "AP50-95": float(result["mAP50-95"] - official["AP"]),
        }
        LOGGER.info(
            f"metric comparison: Trainer AP={result['mAP50-95']:.4f}, AP50={result['mAP50']:.4f} | "
            f"pycocotools AP={official['AP']:.4f}, AP50={official['AP50']:.4f} | "
            f"delta={result['pycoco_delta']['AP50-95']:+.4f}/{result['pycoco_delta']['AP50']:+.4f}"
        )
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


def per_class_metrics(result: dict, names: dict[int, str]) -> dict[str, float]:
    """Return per-class tracker metrics with sanitized metric keys."""
    payload: dict[str, float] = {}
    for c, name in names.items():
        key = name.replace(" ", "_")
        payload[f"val/P_{key}"] = float(result["precision_per_class"][c])
        payload[f"val/R_{key}"] = float(result["recall_per_class"][c])
        payload[f"val/AP50_{key}"] = float(result["ap50_per_class"][c])
        payload[f"val/AP50-95_{key}"] = float(result["ap_per_class"][c])
    return payload


def comparison_metrics(result: dict) -> dict[str, float]:
    """Flatten optional official-COCO comparison values for tracker backends."""
    if "pycoco" not in result:
        return {}
    payload = {f"pycoco/{key}": float(value) for key, value in result["pycoco"].items()}
    payload.update({f"compare/delta_{key}": float(value) for key, value in result["pycoco_delta"].items()})
    return payload


def log_per_class_metrics(tracker, result: dict, names: dict[int, str], step: int) -> None:
    """Log per-class metrics to all tracker backends."""
    tracker.log_metrics(per_class_metrics(result, names), step=step)


def print_val_summary(result: dict, names: dict[int, str]) -> None:
    LOGGER.info(per_class_table(result, names))
