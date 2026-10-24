#!/usr/bin/env python3
"""CPU-capable video/action classifier used by Trainer Automation.

Datasets use ``train/<class>`` and ``val/<class>`` folders, or a compact
``<class>`` layout that Trainer splits deterministically per class.  A small
2-D frame encoder plus GRU keeps the baseline portable while preserving time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from trainer.augmentation import (
    CLASSIFICATION_AUGMENTATION_DEFAULTS,
    add_augmentation_arguments,
    augmentation_config as resolve_augmentation_config,
    augmentation_overrides,
    build_pipeline,
)
from trainer.registry import build_architecture

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def select_device(value: str) -> torch.device:
    value = value.strip().lower()
    if value in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def video_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def class_names(folder: Path) -> list[str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"dataset folder does not exist: {folder}")
    names = sorted(entry.name for entry in folder.iterdir() if entry.is_dir() and video_files(entry))
    if len(names) < 2:
        raise ValueError(f"video classification needs at least two non-empty class folders under {folder}")
    return names


def samples_for(folder: Path, names: list[str]) -> list[Sample]:
    output: list[Sample] = []
    for label, name in enumerate(names):
        target = folder / name
        if target.is_dir():
            output.extend(Sample(path, label) for path in video_files(target))
    return output


def discover_training(root: Path, val_fraction: float, seed: int) -> tuple[list[Sample], list[Sample], list[str]]:
    train_root, val_root = root / "train", root / "val"
    if train_root.is_dir():
        names = class_names(train_root)
        if not val_root.is_dir():
            raise FileNotFoundError(f"validation folder is missing: {val_root}")
        train, val = samples_for(train_root, names), samples_for(val_root, names)
    else:
        names, train, val = class_names(root), [], []
        rng = random.Random(seed)
        for label, name in enumerate(names):
            paths = video_files(root / name)
            rng.shuffle(paths)
            count = max(1, round(len(paths) * val_fraction)) if len(paths) > 1 else 0
            val.extend(Sample(path, label) for path in paths[:count])
            train.extend(Sample(path, label) for path in paths[count:])
    if not train or not val:
        raise ValueError("training and validation splits must both contain videos")
    return train, val, names


def discover_evaluation(root: Path, names: list[str]) -> list[Sample]:
    folder = root / "val" if (root / "val").is_dir() else root
    samples = samples_for(folder, names)
    if not samples:
        raise ValueError(f"no labelled videos found under {folder}")
    return samples


def read_clip(path: Path, clip_len: int, imgsz: int, random_sample: bool, transforms=None) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if total:
        if random_sample and total > clip_len:
            start = random.randint(0, total - clip_len)
            indices = np.linspace(start, min(total - 1, start + clip_len - 1), clip_len).round().astype(int)
        else:
            indices = np.linspace(0, max(total - 1, 0), clip_len).round().astype(int)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
    else:
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if frames:
            indices = np.linspace(0, len(frames) - 1, clip_len).round().astype(int)
            frames = [frames[int(index)] for index in indices]
    capture.release()
    if not frames:
        raise ValueError(f"video has no readable frames: {path}")
    while len(frames) < clip_len:
        frames.append(frames[-1].copy())
    frame_samples = [{"image": frame, "boxes": np.zeros((0, 4), dtype=np.float32),
                      "cls": np.zeros((0,), dtype=np.float32)} for frame in frames[:clip_len]]
    if transforms is not None:
        frame_samples = transforms.apply_many(frame_samples)
    else:
        for frame_sample in frame_samples:
            frame_sample["image"] = cv2.resize(
                frame_sample["image"], (imgsz, imgsz), interpolation=cv2.INTER_AREA,
            )
    tensors = []
    for frame_sample in frame_samples:
        frame = cv2.cvtColor(frame_sample["image"], cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).float().div_(255.0)
        tensors.append(tensor.sub_(0.5).div_(0.25))
    return torch.stack(tensors)


class VideoDataset(Dataset):
    def __init__(self, samples: list[Sample], clip_len: int, imgsz: int, augment: bool = False, *,
                 augmentation: dict | None = None, augmentation_config_path: str | Path | None = None,
                 seed: int = 0):
        self.samples, self.clip_len, self.imgsz, self.augment = samples, clip_len, imgsz, augment
        config = resolve_augmentation_config(
            augmentation, defaults=CLASSIFICATION_AUGMENTATION_DEFAULTS,
            config_path=augmentation_config_path,
        )
        config["imgsz"] = imgsz
        self.transforms = build_pipeline(config, train=augment, seed=seed, allow_multisample=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        return read_clip(sample.path, self.clip_len, self.imgsz, self.augment, self.transforms), sample.label, str(sample.path)


class FrameEncoder(nn.Sequential):
    def __init__(self, width: int):
        layers: list[nn.Module] = []
        previous = 3
        for channels in (width, width * 2, width * 4):
            layers.extend([
                nn.Conv2d(previous, channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels), nn.SiLU(inplace=True),
            ])
            previous = channels
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        super().__init__(*layers)


class TinyVideoClassifier(nn.Module):
    def __init__(self, classes: int, width: int = 24, hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        self.encoder = FrameEncoder(width)
        self.temporal = nn.GRU(width * 4, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, classes))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = clips.shape
        encoded = self.encoder(clips.reshape(batch * time, channels, height, width)).reshape(batch, time, -1)
        sequence, _ = self.temporal(encoded)
        return self.head(sequence[:, -1])


def make_loader(samples: list[Sample], clip_len: int, imgsz: int, batch: int, augment: bool, workers: int, *,
                augmentation: dict | None = None, augmentation_config_path: str | Path | None = None,
                seed: int = 0) -> DataLoader:
    return DataLoader(VideoDataset(
        samples, clip_len, imgsz, augment, augmentation=augmentation,
        augmentation_config_path=augmentation_config_path, seed=seed,
    ), batch_size=batch, shuffle=augment,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, names: list[str]) -> dict:
    model.eval()
    confusion = torch.zeros((len(names), len(names)), dtype=torch.int64)
    loss_fn, total_loss, total = nn.CrossEntropyLoss(reduction="sum"), 0.0, 0
    for clips, labels, _ in loader:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)
        total_loss += float(loss_fn(logits, labels))
        total += labels.numel()
        for actual, predicted in zip(labels.cpu(), logits.argmax(1).cpu(), strict=True):
            confusion[int(actual), int(predicted)] += 1
    per_class = {}
    for index, class_name in enumerate(names):
        support = int(confusion[index].sum())
        per_class[class_name] = {"accuracy": float(confusion[index, index] / support) if support else None, "support": support}
    return {"loss": total_loss / max(total, 1), "accuracy": int(confusion.diag().sum()) / max(total, 1),
            "samples": total, "classes": names, "per_class": per_class, "confusion_matrix": confusion.tolist()}


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def checkpoint_payload(model: nn.Module, names: list[str], args: argparse.Namespace, epoch: int, metrics: dict) -> dict:
    return {"format": "trainer-video-classifier-v2", "model": model.state_dict(), "classes": names,
            "arch": args.arch, "imgsz": args.imgsz, "clip_len": args.clip_len, "width": args.width, "hidden": args.hidden,
            "dropout": args.dropout, "epoch": epoch, "metrics": metrics,
            "augmentation": getattr(args, "_resolved_augmentation", {})}


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") not in {"trainer-video-classifier-v2", "trainer-video-classifier-v1"}:
        raise ValueError(f"not a Trainer video classifier checkpoint: {path}")
    if payload.get("format") == "trainer-video-classifier-v2":
        model = build_architecture("video-classification", str(payload["arch"]), num_classes=len(payload["classes"]),
                                   width=int(payload["width"]), hidden=int(payload["hidden"]),
                                   dropout=float(payload["dropout"]))
    else:
        model = TinyVideoClassifier(len(payload["classes"]), int(payload["width"]), int(payload["hidden"]),
                                    float(payload["dropout"]))
    model.load_state_dict(payload["model"])
    return model.to(device), payload


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_samples, val_samples, names = discover_training(Path(args.data), args.val_fraction, args.seed)
    augmentation = resolve_augmentation_config(
        augmentation_overrides(args), defaults=CLASSIFICATION_AUGMENTATION_DEFAULTS,
        config_path=args.augmentation_config,
    )
    args._resolved_augmentation = augmentation
    train_loader = make_loader(
        train_samples, args.clip_len, args.imgsz, args.batch, True, args.workers,
        augmentation=augmentation, seed=args.seed,
    )
    val_loader = make_loader(val_samples, args.clip_len, args.imgsz, args.batch, False, args.workers, seed=args.seed)
    model = build_architecture("video-classification", args.arch, num_classes=len(names), width=args.width,
                               hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    save_json(output / "classes.json", {str(index): value for index, value in enumerate(names)})
    save_json(output / "augmentation.json", augmentation)
    history, best_accuracy = [], -math.inf
    print(f"video classifier: arch={args.arch}, {len(train_samples)} train, {len(val_samples)} val, "
          f"{len(names)} classes, device={device}")
    for epoch in range(args.epochs):
        model.train(); running_loss = 0.0; seen = 0
        for clips, labels, _ in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(clips); loss = loss_fn(logits, labels)
            loss.backward(); optimizer.step()
            running_loss += float(loss.detach()) * labels.numel(); seen += labels.numel()
        scheduler.step()
        metrics = evaluate_model(model, val_loader, device, names)
        row = {"epoch": epoch + 1, "train_loss": running_loss / max(seen, 1), "val_loss": metrics["loss"],
               "val_accuracy": metrics["accuracy"], "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        torch.save(checkpoint_payload(model, names, args, epoch + 1, metrics), output / "last.pt")
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            torch.save(checkpoint_payload(model, names, args, epoch + 1, metrics), output / "best.pt")
        print(f"epoch {epoch + 1:>4}/{args.epochs} train_loss={row['train_loss']:.4f} val_accuracy={row['val_accuracy']:.4f}")
    save_json(output / "metrics.json", {"best_accuracy": best_accuracy, "history": history, "final": metrics})
    with (output / "history.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    print(f"video classifier artifacts: {output}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    device = select_device(args.device); model, payload = load_checkpoint(Path(args.checkpoint), device)
    samples = discover_evaluation(Path(args.data), list(payload["classes"]))
    loader = make_loader(samples, int(payload["clip_len"]), int(payload["imgsz"]), args.batch, False, args.workers)
    metrics = evaluate_model(model, loader, device, list(payload["classes"])); save_json(Path(args.output), metrics)
    print(json.dumps(metrics, indent=2)); return 0


def source_videos(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        return [source]
    if source.is_dir():
        return video_files(source)
    raise FileNotFoundError(f"video source does not exist: {source}")


@torch.inference_mode()
def predict(args: argparse.Namespace) -> int:
    device = select_device(args.device); model, payload = load_checkpoint(Path(args.checkpoint), device)
    names = list(payload["classes"]); paths = source_videos(Path(args.source))
    if not paths:
        raise ValueError("video source is empty")
    loader = make_loader([Sample(path, 0) for path in paths], int(payload["clip_len"]), int(payload["imgsz"]), args.batch, False, args.workers)
    rows = []; model.eval()
    for clips, _, batch_paths in loader:
        probabilities = model(clips.to(device)).softmax(1).cpu()
        for item, video_path in zip(probabilities, batch_paths, strict=True):
            confidence, index = item.max(0)
            rows.append({"video": video_path, "class": names[int(index)], "confidence": float(confidence)})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "class", "confidence"]); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} video predictions to {output}"); return 0


def export(args: argparse.Namespace) -> int:
    device = select_device(args.device); model, payload = load_checkpoint(Path(args.checkpoint), device); model.eval()
    export_format = str(args.format).lower()
    if export_format not in {"onnx", "tensorrt", "trt", "engine"}:
        raise ValueError("format must be onnx or tensorrt")
    output = Path(args.output)
    stem = output.with_suffix("") if output.suffix.lower() in {".onnx", ".engine"} else output
    onnx_output = stem.with_suffix(".onnx"); onnx_output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, int(payload["clip_len"]), 3, int(payload["imgsz"]), int(payload["imgsz"]), device=device)
    torch.onnx.export(model, dummy, onnx_output, input_names=["clips"], output_names=["logits"], opset_version=args.opset,
                      dynamic_axes={"clips": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    contract = {"format": "trainer-video-classifier-onnx-v1", "classes": payload["classes"],
              "architecture": payload.get("arch", "frame-gru"), "imgsz": payload["imgsz"],
              "clip_len": payload["clip_len"], "input": "clips", "output": "logits", "onnx": str(onnx_output)}
    final_output = onnx_output
    if export_format != "onnx":
        from trainer.detection.engine.exporter import export_tensorrt

        engine_output = stem.with_suffix(".engine")
        final_output = export_tensorrt(
            onnx_output, half=bool(args.half), workspace_gb=int(args.workspace), out_path=engine_output,
        )
        contract.update({"format": "trainer-video-classifier-tensorrt-v1", "engine": str(final_output),
                         "precision": "fp16" if args.half else "fp32"})
    save_json(stem.with_suffix(".json"), contract)
    print(f"exported video classifier: {final_output}"); return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Train/evaluate Trainer video and action classifiers")
    sub = root.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    for command_parser in [train_parser]:
        command_parser.add_argument("--data", required=True); command_parser.add_argument("--output", required=True)
    train_parser.add_argument("--arch", default="r2plus1d-tiny")
    train_parser.add_argument("--epochs", type=int, default=30); train_parser.add_argument("--batch", type=int, default=4)
    train_parser.add_argument("--imgsz", type=int, default=160); train_parser.add_argument("--clip-len", type=int, default=16)
    train_parser.add_argument("--width", type=int, default=24); train_parser.add_argument("--hidden", type=int, default=96)
    train_parser.add_argument("--dropout", type=float, default=0.2); train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4); train_parser.add_argument("--label-smoothing", type=float, default=0.05)
    train_parser.add_argument("--val-fraction", type=float, default=0.2); train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--workers", type=int, default=0); train_parser.add_argument("--device", default="auto")
    add_augmentation_arguments(train_parser)
    evaluate_parser = sub.add_parser("evaluate"); evaluate_parser.add_argument("--data", required=True)
    predict_parser = sub.add_parser("predict"); predict_parser.add_argument("--source", required=True)
    for command_parser in (evaluate_parser, predict_parser):
        command_parser.add_argument("--checkpoint", required=True); command_parser.add_argument("--output", required=True)
        command_parser.add_argument("--batch", type=int, default=4); command_parser.add_argument("--workers", type=int, default=0)
        command_parser.add_argument("--device", default="auto")
    export_parser = sub.add_parser("export"); export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", required=True); export_parser.add_argument("--opset", type=int, default=17)
    export_parser.add_argument("--device", default="cpu")
    export_parser.add_argument("--format", choices=["onnx", "tensorrt", "trt", "engine"], default="onnx")
    export_parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    export_parser.add_argument("--workspace", type=int, default=4)
    return root


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return {"train": train, "evaluate": evaluate, "predict": predict, "export": export}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
