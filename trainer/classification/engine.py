"""Detection-parity lifecycle for Trainer image classification."""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn

from trainer.augmentation import (
    CLASSIFICATION_AUGMENTATION_DEFAULTS,
    add_augmentation_arguments,
    augmentation_config as resolve_augmentation_config,
    augmentation_overrides,
)
from trainer.classification.data import (
    IMAGE_SUFFIXES,
    ImageDataset,
    Sample,
    TinyClassifier,
    classification_mode,
    discover_evaluation,
    discover_training,
    image_files,
    make_loader,
    select_device,
)
from trainer.detection.trackers import build_tracker
from trainer.detection.utils.ema import ModelEMA
from trainer.detection.utils.plotting import make_grid, plot_curve
from trainer.registry import build_architecture

CHECKPOINT_FORMATS = {
    "trainer-image-classifier-v1",
    "trainer-image-classifier-v2",
    "trainer-image-classifier-v3",
    "trainer-image-classifier-v4",
}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_dirs(output: Path) -> dict[str, Path]:
    paths = {
        "root": output,
        "weights": output / "weights",
        "logs": output / "logs",
        "plots": output / "plots",
        "reports": output / "reports",
        "cfg": output / "cfg",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _payload_mode(payload: dict[str, Any]) -> str:
    return str(payload.get("classification_mode", "single-label"))


def _average_precision(actual: np.ndarray, scores: np.ndarray) -> float:
    positives = int(actual.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked = actual[order].astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader, device: torch.device, names: list[str],
                   *, include_probabilities: bool = False, mode: str = "single-label",
                   thresholds: dict[str, float] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    if mode == "multi-label":
        return _evaluate_multilabel(
            model, loader, device, names, include_probabilities=include_probabilities,
            thresholds=thresholds,
        )
    confusion = torch.zeros((len(names), len(names)), dtype=torch.int64)
    total_loss = 0.0
    total = 0
    rows: list[dict[str, Any]] = []
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for images, labels, paths in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probabilities = logits.softmax(1)
        total_loss += float(loss_fn(logits, labels))
        confidence, predictions = probabilities.max(1)
        total += labels.numel()
        for position, (actual, predicted) in enumerate(zip(labels.cpu(), predictions.cpu(), strict=True)):
            actual_index, predicted_index = int(actual), int(predicted)
            confusion[actual_index, predicted_index] += 1
            row: dict[str, Any] = {
                "image": str(paths[position]),
                "actual": names[actual_index],
                "predicted": names[predicted_index],
                "confidence": float(confidence[position]),
                "correct": actual_index == predicted_index,
            }
            if include_probabilities:
                row["probabilities"] = [float(value) for value in probabilities[position].cpu()]
                row["actual_index"] = actual_index
                row["predicted_index"] = predicted_index
            rows.append(row)

    matrix = confusion.numpy()
    per_class: dict[str, dict[str, float | int]] = {}
    precisions, recalls, f1s = [], [], []
    for index, name in enumerate(names):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        tn = int(matrix.sum() - tp - fp - fn)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        per_class[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": recall, "support": int(matrix[index].sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    correct = int(np.trace(matrix))
    metrics = {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "samples": total,
        "classes": names,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, rows


@torch.inference_mode()
def _evaluate_multilabel(model: nn.Module, loader, device: torch.device, names: list[str],
                         *, include_probabilities: bool = False,
                         thresholds: dict[str, float] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    threshold_values = torch.tensor(
        [float((thresholds or {}).get(name, 0.5)) for name in names], device=device,
    )
    total_loss = 0.0
    total = exact_matches = 0
    all_targets: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for images, targets, paths in loader:
        images, targets = images.to(device), targets.float().to(device)
        logits = model(images)
        probabilities = logits.sigmoid()
        total_loss += float(nn.functional.binary_cross_entropy_with_logits(logits, targets)) * images.shape[0]
        predictions = probabilities >= threshold_values
        actual = targets >= 0.5
        exact_matches += int((predictions == actual).all(dim=1).sum())
        total += images.shape[0]
        all_targets.append(actual.cpu())
        all_probabilities.append(probabilities.cpu())
        for position in range(images.shape[0]):
            actual_indices = torch.where(actual[position])[0].cpu().tolist()
            predicted_indices = torch.where(predictions[position])[0].cpu().tolist()
            row: dict[str, Any] = {
                "image": str(paths[position]),
                "actual": json.dumps([names[index] for index in actual_indices]),
                "predicted": json.dumps([names[index] for index in predicted_indices]),
                "confidence": float(probabilities[position].max()),
                "correct": actual_indices == predicted_indices,
            }
            if include_probabilities:
                row["probabilities"] = [float(value) for value in probabilities[position].cpu()]
                row["actual_indices"] = actual_indices
                row["predicted_indices"] = predicted_indices
            rows.append(row)

    target_matrix = torch.cat(all_targets).numpy().astype(bool)
    probability_matrix = torch.cat(all_probabilities).numpy()
    prediction_matrix = probability_matrix >= threshold_values.cpu().numpy()[None, :]
    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    aps: list[float] = []
    total_tp = total_fp = total_fn = 0
    for index, name in enumerate(names):
        actual = target_matrix[:, index]
        predicted = prediction_matrix[:, index]
        tp = int((actual & predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        tn = int((~actual & ~predicted).sum())
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        ap = _average_precision(actual, probability_matrix[:, index])
        per_class[name] = {
            "precision": precision, "recall": recall, "f1": f1, "ap": ap,
            "accuracy": _safe_ratio(tp + tn, total), "support": int(actual.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        precisions.append(precision); recalls.append(recall); f1s.append(f1)
        if actual.any():
            aps.append(ap)
        total_tp += tp; total_fp += fp; total_fn += fn
    micro_precision = _safe_ratio(total_tp, total_tp + total_fp)
    micro_recall = _safe_ratio(total_tp, total_tp + total_fn)
    metrics = {
        "classification_mode": "multi-label",
        "loss": total_loss / max(total, 1),
        "accuracy": exact_matches / max(total, 1),
        "subset_accuracy": exact_matches / max(total, 1),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": _safe_ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "map": float(np.mean(aps)) if aps else 0.0,
        "samples": total,
        "classes": names,
        "per_class": per_class,
        "confusion_matrix": [],
    }
    return metrics, rows


def _build_from_payload(payload: dict[str, Any], device: torch.device) -> nn.Module:
    names = list(payload["classes"])
    if payload.get("format") in {"trainer-image-classifier-v2", "trainer-image-classifier-v3",
                                  "trainer-image-classifier-v4"}:
        model = build_architecture(
            "classification", str(payload.get("arch", "convnext-tiny")), num_classes=len(names),
            width=int(payload.get("width", 32)), dropout=float(payload.get("dropout", 0.15)),
        )
    else:
        model = TinyClassifier(len(names), int(payload.get("width", 32)), float(payload.get("dropout", 0.15)))
    model.load_state_dict(payload["model"])
    return model.to(device)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") not in CHECKPOINT_FORMATS:
        raise ValueError(f"not a Trainer image classifier checkpoint: {path}")
    return _build_from_payload(payload, device), payload


def _transfer_weights(model: nn.Module, state: dict[str, torch.Tensor]) -> tuple[int, int]:
    current = model.state_dict()
    selected = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
    model.load_state_dict(selected, strict=False)
    return len(selected), len(state) - len(selected)


def _freeze_features(model: nn.Module, count: int) -> int:
    if count <= 0:
        return 0
    modules: list[nn.Module] = []
    if hasattr(model, "downsample") and hasattr(model, "stages"):
        for downsample, stage in zip(model.downsample, model.stages, strict=True):
            modules.extend([downsample, stage])
    elif hasattr(model, "features"):
        modules.extend(list(model.features.children()))
    else:
        modules.extend(list(model.children())[:-1])
    frozen = 0
    for module in modules[:count]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        frozen += 1
    return frozen


def _optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("freeze setting left no trainable classifier parameters")
    if args.optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay, nesterov=True)
    return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)


def _classification_loss(logits: torch.Tensor, targets: torch.Tensor, smoothing: float,
                         mode: str = "single-label") -> torch.Tensor:
    """Cross entropy for hard labels or augmentation-produced soft targets."""
    if mode == "multi-label":
        targets = targets.float()
        if smoothing:
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)
    if targets.ndim == 1:
        return nn.functional.cross_entropy(logits, targets, label_smoothing=smoothing)
    if smoothing:
        targets = targets * (1.0 - smoothing) + smoothing / targets.shape[1]
    return -(targets * nn.functional.log_softmax(logits, dim=1)).sum(1).mean()


def _checkpoint_payload(model: nn.Module, ema: ModelEMA, names: list[str], args: argparse.Namespace,
                        epoch: int, metrics: dict[str, Any], history: list[dict[str, Any]],
                        best_accuracy: float, optimizer, scheduler, scaler, mode: str) -> dict[str, Any]:
    return {
        "format": "trainer-image-classifier-v4",
        "model": ema.ema.state_dict(),
        "raw_model": ModelEMA._unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "ema_state": ema.state_dict(),
        "classes": names,
        "classification_mode": mode,
        "activation": "sigmoid" if mode == "multi-label" else "softmax",
        "arch": args.arch,
        "imgsz": args.imgsz,
        "width": args.width,
        "dropout": args.dropout,
        "epoch": epoch,
        "metrics": metrics,
        "history": history,
        "best_accuracy": best_accuracy,
        "args": vars(args),
    }


def _save_checkpoint(payload: dict[str, Any], paths: dict[str, Path], *, best: bool,
                     save_period: int) -> None:
    torch.save(payload, paths["weights"] / "last.pt")
    torch.save(payload, paths["root"] / "last.pt")
    if best:
        torch.save(payload, paths["weights"] / "best.pt")
        torch.save(payload, paths["root"] / "best.pt")
    epoch = int(payload["epoch"])
    if save_period > 0 and epoch % save_period == 0:
        torch.save(payload, paths["weights"] / f"epoch_{epoch}.pt")


def _plot_confusion(matrix: list[list[int]], names: list[str], output: Path) -> Path:
    values = np.asarray(matrix, dtype=np.float32)
    size = max(420, 85 * len(names) + 180)
    canvas = np.full((size, size, 3), 255, np.uint8)
    left, top, right, bottom = 120, 50, 40, 110
    cell_w = max(1, (size - left - right) // len(names))
    cell_h = max(1, (size - top - bottom) // len(names))
    maximum = max(float(values.max()), 1.0)
    for row in range(len(names)):
        for column in range(len(names)):
            intensity = int(255 - 200 * values[row, column] / maximum)
            x0, y0 = left + column * cell_w, top + row * cell_h
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_w, y0 + cell_h), (255, intensity, intensity), -1)
            text = str(int(values[row, column]))
            cv2.putText(canvas, text, (x0 + 8, y0 + cell_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, names[row][:14], (5, top + row * cell_h + cell_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
    for column, name in enumerate(names):
        cv2.putText(canvas, name[:12], (left + column * cell_w, size - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Predicted class", (left, size - 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Actual / predicted confusion matrix", (left, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return output


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_report(run_dir: Path, metrics: dict[str, Any], history: list[dict[str, Any]],
                 model_name: str, dataset: str, *, output: Path | None = None,
                 thresholds: dict[str, Any] | None = None) -> Path:
    run_dir = Path(run_dir)
    paths = _run_dirs(run_dir)
    multi_label = metrics.get("classification_mode") == "multi-label"
    confusion_section = ""
    if metrics.get("confusion_matrix"):
        _plot_confusion(metrics["confusion_matrix"], list(metrics["classes"]), paths["plots"] / "confusion_matrix.png")
        confusion_section = '<h2>Confusion matrix</h2><img src="../plots/confusion_matrix.png" alt="Confusion matrix">'
    if history:
        x = np.array([float(row["epoch"]) for row in history])
        plot_curve(x, {
            "train loss": np.array([float(row["train_loss"]) for row in history]),
            "validation loss": np.array([float(row["val_loss"]) for row in history]),
        }, paths["plots"] / "loss_curve.png", title="Classification loss", xlabel="Epoch", ylabel="Loss")
        plot_curve(x, {"validation accuracy": np.array([float(row["val_accuracy"]) for row in history])},
                   paths["plots"] / "accuracy_curve.png", title="Classification accuracy",
                   xlabel="Epoch", ylabel="Accuracy")

    summary = [("Exact-match accuracy" if multi_label else "Accuracy", metrics["accuracy"])]
    if multi_label:
        summary.extend([("mAP", metrics["map"]), ("Micro F1", metrics["micro_f1"])])
    summary.extend([
        ("Macro precision", metrics["macro_precision"]), ("Macro recall", metrics["macro_recall"]),
        ("Macro F1", metrics["macro_f1"]), ("Loss", metrics["loss"]), ("Samples", metrics["samples"]),
    ])
    class_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            name, f"{values['precision']:.4f}", f"{values['recall']:.4f}", f"{values['f1']:.4f}",
            *([f"{values.get('ap', 0.0):.4f}"] if multi_label else []),
            values["support"], values["tp"], values["fp"], values["fn"], values["tn"],
        )) + "</tr>"
        for name, values in metrics["per_class"].items()
    )
    threshold_section = ""
    if thresholds:
        threshold_rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{value:.4f}</td></tr>"
            for name, value in thresholds.get("thresholds", {}).items()
        )
        threshold_section = (
            "<h2>Per-class confidence thresholds</h2>"
            f"<table><tr><th>Class</th><th>Threshold</th></tr>{threshold_rows}</table>"
        )
    summary_rows = "".join(f"<tr><th>{label}</th><td>{value:.4f}</td></tr>" if isinstance(value, float)
                           else f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in summary)
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Trainer classification report</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:36px auto;color:#18212f}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #d7dde5;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;border:1px solid #d7dde5}}code{{background:#f2f4f7;padding:2px 5px}}</style></head><body>
<h1>Trainer image classification report</h1><p>Generated {datetime.now().isoformat(timespec='minutes')}</p>
<p><strong>Architecture:</strong> <code>{html.escape(model_name)}</code><br><strong>Dataset:</strong> <code>{html.escape(dataset)}</code><br><strong>Target contract:</strong> <code>{'multi-label' if multi_label else 'single-label'}</code></p>
<h2>Summary</h2><table>{summary_rows}</table>
<h2>Per-class metrics</h2><table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th>{'<th>AP</th>' if multi_label else ''}<th>Support</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr>{class_rows}</table>
{threshold_section}{confusion_section}
{('<h2>Training curves</h2><img src="../plots/loss_curve.png" alt="Loss curve"><img src="../plots/accuracy_curve.png" alt="Accuracy curve">' if history else '')}
</body></html>"""
    destination = output or paths["reports"] / "report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf8")
    return destination


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    _seed(args.seed)
    device = select_device(args.device)
    output = Path(args.output)
    paths = _run_dirs(output)
    train_samples, val_samples, names = discover_training(Path(args.data), args.val_fraction, args.seed)
    mode = classification_mode(train_samples + val_samples)
    save_json(paths["root"] / "classes.json", {str(index): value for index, value in enumerate(names)})

    resume_payload: dict[str, Any] | None = None
    resume_path: Path | None = None
    if getattr(args, "resume", ""):
        resume_path = paths["weights"] / "last.pt" if args.resume == "auto" else Path(args.resume)
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if list(resume_payload.get("classes", [])) != names:
            raise ValueError("resume dataset classes do not match checkpoint classes")
        if _payload_mode(resume_payload) != mode:
            raise ValueError("resume dataset target contract does not match checkpoint classification mode")
        args.arch = str(resume_payload.get("arch", args.arch))
        args.width = int(resume_payload.get("width", args.width))
        args.dropout = float(resume_payload.get("dropout", args.dropout))
        args.imgsz = int(resume_payload.get("imgsz", args.imgsz))

    source_payload: dict[str, Any] | None = None
    if args.command == "finetune":
        source_payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
        if source_payload.get("format") not in CHECKPOINT_FORMATS:
            raise ValueError("finetune checkpoint is not a Trainer classifier")
        args.arch = args.arch or str(source_payload.get("arch", "convnext-tiny"))
        args.width = int(source_payload.get("width", 32)) if args.width is None else args.width
        args.dropout = float(source_payload.get("dropout", 0.15)) if args.dropout is None else args.dropout
        args.imgsz = int(source_payload.get("imgsz", 160)) if args.imgsz is None else args.imgsz

    augmentation = resolve_augmentation_config(
        augmentation_overrides(args), defaults=CLASSIFICATION_AUGMENTATION_DEFAULTS,
        config_path=args.augmentation_config,
    )
    save_json(paths["cfg"] / "args.json", vars(args))
    save_json(paths["cfg"] / "augmentation.json", augmentation)

    model = build_architecture("classification", args.arch, num_classes=len(names), width=args.width,
                               dropout=args.dropout).to(device)
    if resume_payload:
        model.load_state_dict(resume_payload.get("raw_model", resume_payload["model"]))
    elif source_payload:
        loaded, skipped = _transfer_weights(model, source_payload["model"])
        print(f"finetune transfer: loaded={loaded}, skipped={skipped} tensors")
    frozen = _freeze_features(model, args.freeze)
    if frozen:
        print(f"froze {frozen} feature module(s)")
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)

    train_loader = make_loader(
        train_samples, args.imgsz, args.batch, True, args.workers, num_classes=len(names),
        augmentation=augmentation, seed=args.seed,
    )
    val_loader = make_loader(
        val_samples, args.imgsz, args.batch, False, args.workers, num_classes=len(names), seed=args.seed,
    )
    optimizer = _optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(args.epochs, 1), eta_min=args.lr * args.final_lr_fraction,
    )
    amp_enabled = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    ema = ModelEMA(model, decay_max=args.ema_decay, warmup_steps=args.ema_warmup_steps)
    history: list[dict[str, Any]] = []
    best_accuracy = -math.inf
    start_epoch = 0
    no_improve = 0

    if resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        scaler.load_state_dict(resume_payload.get("scaler", {}))
        if resume_payload.get("ema_state"):
            ema.load_state_dict(resume_payload["ema_state"])
        history = list(resume_payload.get("history", []))
        best_accuracy = float(resume_payload.get("best_accuracy", -math.inf))
        start_epoch = int(resume_payload["epoch"])
        print(f"resuming {resume_path} from epoch {start_epoch + 1}")
    if start_epoch >= args.epochs:
        raise ValueError(f"checkpoint already reached epoch {start_epoch}; set epochs higher than {start_epoch}")

    tracker_cfg = {
        "mlflow": args.mlflow, "mlflow_tracking_uri": args.mlflow_tracking_uri,
        "mlflow_experiment": args.mlflow_experiment, "mlflow_run_name": args.mlflow_run_name,
    }
    tracker = build_tracker(tracker_cfg, output)
    tracker.log_params(vars(args))
    tracker.set_tags({"task": "classification", "architecture": args.arch, "dataset": args.data,
                      "classification_mode": mode})
    print(f"classifier: arch={args.arch}, {len(train_samples)} train, {len(val_samples)} val, "
          f"{len(names)} classes, mode={mode}, device={device}, run={output}")

    metrics: dict[str, Any] = {}
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        for batch_index, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            if args.channels_last and device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = _classification_loss(logits, labels, args.label_smoothing, mode)
            scaler.scale(loss / args.accumulate).backward()
            should_step = (batch_index + 1) % args.accumulate == 0 or batch_index + 1 == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            running_loss += float(loss.detach()) * images.shape[0]
            seen += images.shape[0]
        scheduler.step()
        metrics, _ = evaluate_model(ema.ema, val_loader, device, names, mode=mode)
        row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_map": metrics.get("map", metrics["macro_f1"]),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        selection_score = metrics.get("map", metrics["accuracy"])
        improved = selection_score > best_accuracy + args.min_delta
        if improved:
            best_accuracy = float(selection_score)
            no_improve = 0
        else:
            no_improve += 1
        payload = _checkpoint_payload(model, ema, names, args, epoch + 1, metrics, history,
                                      best_accuracy, optimizer, scheduler, scaler, mode)
        payload["augmentation"] = augmentation
        _save_checkpoint(payload, paths, best=improved, save_period=args.save_period)
        save_json(paths["root"] / "metrics.json", {
            "best_score": best_accuracy, "selection_metric": "map" if mode == "multi-label" else "accuracy",
            "best_accuracy": best_accuracy, "history": history, "final": metrics,
        })
        _write_csv(paths["logs"] / "history.csv", history)
        tracker.log_metrics({
            "train/loss": row["train_loss"], "val/loss": row["val_loss"],
            "val/accuracy": row["val_accuracy"], "val/macro_f1": row["val_macro_f1"],
            "val/map": row["val_map"],
            "lr": row["lr"], "epoch_time": row["epoch_seconds"],
        }, step=epoch)
        print(f"epoch {epoch + 1:>4}/{args.epochs} train_loss={row['train_loss']:.4f} "
              f"val_loss={row['val_loss']:.4f} accuracy={row['val_accuracy']:.4f} "
              f"macro_f1={row['val_macro_f1']:.4f}"
              f"{' map=' + format(row['val_map'], '.4f') if mode == 'multi-label' else ''}")
        if epoch + 1 >= args.min_epochs and no_improve >= args.patience:
            print(f"early stopping at epoch {epoch + 1}; no improvement for {no_improve} epoch(s)")
            break

    write_report(output, metrics, history, args.arch, args.data)
    tracker.log_artifact(paths["weights"], "weights")
    tracker.log_artifact(paths["reports"], "reports")
    tracker.finish()
    print(f"classifier artifacts: {output}")
    return metrics


def train(args: argparse.Namespace) -> int:
    run_training(args)
    return 0


def evaluate(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    model, payload = load_checkpoint(Path(args.checkpoint), device)
    names = list(payload["classes"])
    mode = _payload_mode(payload)
    samples = discover_evaluation(Path(args.data), names, mode)
    loader = make_loader(samples, int(payload["imgsz"]), args.batch, False, args.workers,
                         num_classes=len(names))
    metrics, rows = evaluate_model(model, loader, device, names, mode=mode)
    output = Path(args.output)
    save_json(output, metrics)
    public_rows = [{key: value for key, value in row.items() if key not in {"probabilities", "actual_index", "predicted_index"}}
                   for row in rows]
    _write_csv(output.with_name("per_image.csv"), public_rows)
    if metrics.get("confusion_matrix"):
        confusion_rows = [{"actual": name, **{predicted: metrics["confusion_matrix"][index][column]
                           for column, predicted in enumerate(names)}} for index, name in enumerate(names)]
        _write_csv(output.with_name("confusion_matrix.csv"), confusion_rows)
    write_report(output.parent, metrics, [], str(payload.get("arch", "classifier")), args.data)
    print(json.dumps(metrics, indent=2))
    return 0


def source_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return image_files(source)
    raise FileNotFoundError(f"image source does not exist: {source}")


@torch.inference_mode()
def predict(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    model, payload = load_checkpoint(Path(args.checkpoint), device)
    names = list(payload["classes"])
    mode = _payload_mode(payload)
    samples = [Sample(path, 0) for path in source_images(Path(args.source))]
    if not samples:
        raise ValueError("image source is empty")
    loader = make_loader(samples, int(payload["imgsz"]), args.batch, False, args.workers,
                         num_classes=len(names))
    thresholds: dict[str, float] = {}
    if args.thresholds:
        thresholds = {str(key): float(value) for key, value in json.loads(Path(args.thresholds).read_text())["thresholds"].items()}
    rows: list[dict[str, Any]] = []
    model.eval()
    for images, _, paths in loader:
        logits = model(images.to(device))
        probabilities = (logits.sigmoid() if mode == "multi-label" else logits.softmax(1)).cpu()
        top_values, top_indices = probabilities.topk(min(args.topk, len(names)), dim=1)
        for row_index, (values, indices, image_path) in enumerate(zip(top_values, top_indices, paths, strict=True)):
            best_name = names[int(indices[0])]
            best_confidence = float(values[0])
            default_conf = args.conf if args.conf > 0 else (0.5 if mode == "multi-label" else 0.0)
            accepted_indices = [index for index, probability in enumerate(probabilities[row_index])
                                if float(probability) >= thresholds.get(names[index], default_conf)]
            accepted = bool(accepted_indices) if mode == "multi-label" else best_confidence >= thresholds.get(best_name, default_conf)
            rows.append({
                "image": image_path,
                "class": best_name if accepted else "unknown",
                "confidence": best_confidence,
                "accepted": accepted,
                "classes": json.dumps([names[index] for index in accepted_indices]) if mode == "multi-label" else json.dumps([best_name] if accepted else []),
                "topk": json.dumps([{"class": names[int(index)], "confidence": float(value)}
                                     for value, index in zip(values, indices, strict=True)]),
            })
    _write_csv(Path(args.output), rows)
    print(f"wrote {len(rows)} predictions to {args.output}")
    return 0


def xai(args: argparse.Namespace) -> int:
    from trainer.classification.xai import generate_xai

    device = select_device(args.device)
    model, payload = load_checkpoint(Path(args.checkpoint), device)
    result = generate_xai(
        model, payload, source_images(Path(args.source)), Path(args.output), device,
        target=args.target, max_images=args.max_images, intensity=args.intensity,
        activation_channels=args.activation_channels,
    )
    print(json.dumps({
        "architecture": result["architecture"], "images": len(result["images"]),
        "gallery": result["gallery"], "method": result["method"],
    }, indent=2))
    return 0


def export(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    model, payload = load_checkpoint(Path(args.checkpoint), device)
    model.eval()
    export_format = str(args.format).lower()
    if export_format not in {"onnx", "tensorrt", "trt", "engine"}:
        raise ValueError("format must be onnx or tensorrt")
    output = Path(args.output)
    stem = output.with_suffix("") if output.suffix.lower() in {".onnx", ".engine"} else output
    onnx_output = stem.with_suffix(".onnx")
    onnx_output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, int(payload["imgsz"]), int(payload["imgsz"]), device=device)
    torch.onnx.export(
        model, dummy, onnx_output, input_names=["images"], output_names=["logits"],
        opset_version=args.opset, dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False,
    )
    contract = {
        "format": "trainer-image-classifier-onnx-v1", "classes": payload["classes"],
        "architecture": payload.get("arch", "tiny-cnn"), "imgsz": payload["imgsz"],
        "normalization": {"scale": 255.0, "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
        "input": "images", "output": "logits", "onnx": str(onnx_output),
        "classification_mode": _payload_mode(payload),
        "activation": "sigmoid" if _payload_mode(payload) == "multi-label" else "softmax",
        "default_threshold": 0.5 if _payload_mode(payload) == "multi-label" else None,
    }
    final_output = onnx_output
    if export_format != "onnx":
        from trainer.detection.engine.exporter import export_tensorrt

        engine_output = stem.with_suffix(".engine")
        final_output = export_tensorrt(
            onnx_output, half=bool(args.half), workspace_gb=int(args.workspace), out_path=engine_output,
        )
        contract.update({"format": "trainer-image-classifier-tensorrt-v1", "engine": str(final_output),
                         "precision": "fp16" if args.half else "fp32"})
    save_json(stem.with_suffix(".json"), contract)
    print(f"exported classifier: {final_output}")
    return 0


def tune_thresholds(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    model, payload = load_checkpoint(Path(args.checkpoint), device)
    names = list(payload["classes"])
    mode = _payload_mode(payload)
    samples = discover_evaluation(Path(args.data), names, mode)
    loader = make_loader(samples, int(payload["imgsz"]), args.batch, False, args.workers,
                         num_classes=len(names))
    metrics, rows = evaluate_model(model, loader, device, names, include_probabilities=True, mode=mode)
    grid = np.linspace(0.01, 0.99, args.steps)
    thresholds: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float | int]] = {}
    for class_index, name in enumerate(names):
        actual = np.array([
            class_index in row["actual_indices"] if mode == "multi-label" else row["actual_index"] == class_index
            for row in rows
        ])
        predicted = np.ones(len(rows), dtype=bool) if mode == "multi-label" else np.array([
            row["predicted_index"] == class_index for row in rows
        ])
        confidence = np.array([row["probabilities"][class_index] for row in rows])
        scores = []
        for threshold in grid:
            positive = predicted & (confidence >= threshold)
            tp = int((positive & actual).sum())
            fp = int((positive & ~actual).sum())
            fn = int((~positive & actual).sum())
            precision = _safe_ratio(tp, tp + fp)
            recall = _safe_ratio(tp, tp + fn)
            f1 = _safe_ratio(2 * precision * recall, precision + recall)
            scores.append((f1, precision, recall, threshold))
        f1, precision, recall, threshold = max(scores, key=lambda item: item[0])
        thresholds[name] = round(float(threshold), 4)
        diagnostics[name] = {"f1": f1, "precision": precision, "recall": recall,
                             "threshold": thresholds[name], "support": int(actual.sum())}
    result = {"thresholds": thresholds, "diagnostics": diagnostics, "metrics": metrics}
    save_json(Path(args.output), result)
    print(json.dumps({"thresholds": thresholds, "diagnostics": diagnostics}, indent=2))
    return 0


def report(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    model, payload = load_checkpoint(Path(args.checkpoint), device)
    names = list(payload["classes"])
    mode = _payload_mode(payload)
    samples = discover_evaluation(Path(args.data), names, mode)
    loader = make_loader(samples, int(payload["imgsz"]), args.batch, False, args.workers,
                         num_classes=len(names))
    threshold_payload = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    metrics, _ = evaluate_model(
        model, loader, device, names, mode=mode,
        thresholds=threshold_payload.get("thresholds") if threshold_payload else None,
    )
    checkpoint = Path(args.checkpoint)
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "weights" else checkpoint.parent
    history = list(payload.get("history", [])) or _read_history(run_dir / "logs" / "history.csv")
    thresholds = threshold_payload
    destination = Path(args.output)
    write_report(destination.parent.parent if destination.parent.name == "reports" else destination.parent,
                 metrics, history, str(payload.get("arch", "classifier")), args.data,
                 output=destination, thresholds=thresholds)
    print(f"classification report: {destination}")
    return 0


def viz_augment(args: argparse.Namespace) -> int:
    _seed(args.seed)
    train_samples, _, names = discover_training(Path(args.data), args.val_fraction, args.seed)
    augmentation = resolve_augmentation_config(
        augmentation_overrides(args), defaults=CLASSIFICATION_AUGMENTATION_DEFAULTS,
        config_path=args.augmentation_config,
    )
    dataset = ImageDataset(
        train_samples, args.imgsz, augment=True, num_classes=len(names), augmentation=augmentation,
        seed=args.seed,
    )
    images: list[np.ndarray] = []
    for index in range(min(args.n, len(dataset))):
        tensor, label, _ = dataset[index]
        if isinstance(label, torch.Tensor):
            label_indices = torch.where(label >= 0.5)[0].tolist()
            label_text = ", ".join(names[item] for item in label_indices[:4]) or "no positive labels"
        else:
            label_text = names[int(label)]
        rgb = np.clip((tensor.numpy().transpose(1, 2, 0) * 0.25 + 0.5) * 255, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (0, 0), (image.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(image, label_text[:54], (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)
        images.append(image)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), make_grid(images, cols=min(4, len(images))))
    print(f"augmentation preview: {output}")
    return 0


def bench_data(args: argparse.Namespace) -> int:
    train_samples, _, names = discover_training(Path(args.data), args.val_fraction, args.seed)
    augmentation = resolve_augmentation_config(
        augmentation_overrides(args), defaults=CLASSIFICATION_AUGMENTATION_DEFAULTS,
        config_path=args.augmentation_config,
    )
    loader = make_loader(
        train_samples, args.imgsz, args.batch, True, args.workers, num_classes=len(names),
        augmentation=augmentation, seed=args.seed,
    )
    started = time.perf_counter()
    images = 0
    batches = 0
    for batch_images, _, _ in loader:
        images += len(batch_images)
        batches += 1
        if batches >= args.batches:
            break
    elapsed = time.perf_counter() - started
    result = {"images": images, "batches": batches, "seconds": elapsed,
              "images_per_second": images / max(elapsed, 1e-9), "workers": args.workers,
              "batch_size": args.batch, "imgsz": args.imgsz}
    save_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    return 0


def tune(args: argparse.Namespace) -> int:
    _seed(args.seed)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for index in range(args.iterations):
        trial = copy.deepcopy(args)
        trial.command = "train"
        trial.output = str(root / f"trial_{index + 1}")
        trial.epochs = args.tune_epochs
        trial.resume = ""
        trial.patience = args.tune_epochs + 1
        trial.min_epochs = args.tune_epochs + 1
        trial.lr = 10 ** rng.uniform(-4.2, -2.3)
        trial.weight_decay = 10 ** rng.uniform(-6.0, -2.5)
        trial.dropout = rng.uniform(0.0, 0.35)
        trial.label_smoothing = rng.uniform(0.0, 0.15)
        metrics = run_training(trial)
        score = metrics.get("map", metrics["accuracy"])
        value = {
            "trial": index + 1, "accuracy": metrics["accuracy"], "map": metrics.get("map"),
            "score": score, "macro_f1": metrics["macro_f1"],
            "lr": trial.lr, "weight_decay": trial.weight_decay, "dropout": trial.dropout,
            "label_smoothing": trial.label_smoothing, "run": trial.output,
        }
        trials.append(value)
        if best is None or value["score"] > best["score"]:
            best = value
        save_json(root / "tune_results.json", {"best": best, "trials": trials})
        print(f"tune trial {index + 1}/{args.iterations}: score={value['score']:.4f}")
    save_json(root / "best_hyp.json", {key: best[key] for key in
              ("lr", "weight_decay", "dropout", "label_smoothing", "score", "accuracy", "map", "macro_f1")})
    return 0


def _add_training_arguments(parser: argparse.ArgumentParser, *, arch_default: str | None = "convnext-tiny") -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch", default=arch_default)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--final-lr-fraction", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-warmup-steps", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--resume", default="")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mlflow-tracking-uri", default="")
    parser.add_argument("--mlflow-experiment", default="trainer-classification")
    parser.add_argument("--mlflow-run-name", default="classification")
    add_augmentation_arguments(parser)


def _add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Trainer image classification lifecycle")
    sub = root.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    _add_training_arguments(train_parser)

    finetune_parser = sub.add_parser("finetune")
    _add_training_arguments(finetune_parser, arch_default=None)
    finetune_parser.set_defaults(width=None, dropout=None, imgsz=None)
    finetune_parser.add_argument("--checkpoint", required=True)

    evaluate_parser = sub.add_parser("evaluate", aliases=["val"])
    _add_eval_arguments(evaluate_parser)

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--batch", type=int, default=32)
    predict_parser.add_argument("--workers", type=int, default=0)
    predict_parser.add_argument("--device", default="auto")
    predict_parser.add_argument("--conf", type=float, default=0.0)
    predict_parser.add_argument("--thresholds", default="")
    predict_parser.add_argument("--topk", type=int, default=3)

    xai_parser = sub.add_parser("xai")
    xai_parser.add_argument("--source", required=True)
    xai_parser.add_argument("--checkpoint", required=True)
    xai_parser.add_argument("--output", required=True)
    xai_parser.add_argument("--device", default="auto")
    xai_parser.add_argument("--target", default="predicted")
    xai_parser.add_argument("--max-images", type=int, default=24)
    xai_parser.add_argument("--intensity", type=float, default=0.72)
    xai_parser.add_argument("--activation-channels", type=int, default=4)

    export_parser = sub.add_parser("export")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--format", choices=["onnx", "tensorrt", "trt", "engine"], default="onnx")
    export_parser.add_argument("--opset", type=int, default=17)
    export_parser.add_argument("--device", default="cpu")
    export_parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    export_parser.add_argument("--workspace", type=int, default=4)

    threshold_parser = sub.add_parser("tune-thresh")
    _add_eval_arguments(threshold_parser)
    threshold_parser.add_argument("--steps", type=int, default=200)

    report_parser = sub.add_parser("report")
    _add_eval_arguments(report_parser)
    report_parser.add_argument("--thresholds", default="")

    viz_parser = sub.add_parser("viz-augment")
    viz_parser.add_argument("--data", required=True)
    viz_parser.add_argument("--output", required=True)
    viz_parser.add_argument("--imgsz", type=int, default=160)
    viz_parser.add_argument("--n", type=int, default=16)
    viz_parser.add_argument("--val-fraction", type=float, default=0.2)
    viz_parser.add_argument("--seed", type=int, default=0)
    add_augmentation_arguments(viz_parser)

    bench_parser = sub.add_parser("bench-data")
    bench_parser.add_argument("--data", required=True)
    bench_parser.add_argument("--output", required=True)
    bench_parser.add_argument("--imgsz", type=int, default=160)
    bench_parser.add_argument("--batch", type=int, default=16)
    bench_parser.add_argument("--workers", type=int, default=0)
    bench_parser.add_argument("--batches", type=int, default=50)
    bench_parser.add_argument("--val-fraction", type=float, default=0.2)
    bench_parser.add_argument("--seed", type=int, default=0)
    add_augmentation_arguments(bench_parser)

    tune_parser = sub.add_parser("tune")
    _add_training_arguments(tune_parser)
    tune_parser.add_argument("--iterations", type=int, default=30)
    tune_parser.add_argument("--tune-epochs", type=int, default=5)
    return root


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    handlers = {
        "train": train, "finetune": train, "evaluate": evaluate, "val": evaluate,
        "predict": predict, "xai": xai, "export": export, "tune": tune, "tune-thresh": tune_thresholds,
        "report": report, "viz-augment": viz_augment, "bench-data": bench_data,
    }
    return handlers[args.command](args)


__all__ = [
    "bench_data", "evaluate", "evaluate_model", "export", "load_checkpoint", "main", "predict", "xai",
    "report", "run_training", "train", "tune", "tune_thresholds", "viz_augment", "write_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
