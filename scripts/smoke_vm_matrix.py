#!/usr/bin/env python3
"""Train every bundled model family on tiny synthetic data.

This is deliberately small enough for a CPU VM while still exercising real
forward, backward, optimizer, checkpoint, and artifact paths.  The VLM smoke
uses Trainer's offline tiny LoRA architecture so the gate never depends on a
model download.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def image_classification_data(root: Path, size: int = 48) -> Path:
    rng = np.random.default_rng(11)
    for split, count in (("train", 4), ("val", 2)):
        for name, value in (("dark", 35), ("light", 220)):
            folder = root / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image = np.full((size, size, 3), value, dtype=np.uint8)
                noise = rng.integers(0, 16, image.shape, dtype=np.uint8)
                image = cv2.add(image, noise) if name == "dark" else cv2.subtract(image, noise)
                cv2.circle(image, (12 + index * 3, 24), 7, (255 - value,) * 3, -1)
                if not cv2.imwrite(str(folder / f"{index:02d}.png"), image):
                    raise RuntimeError("could not write classification fixture")
    return root


def detection_data(root: Path, size: int = 64) -> Path:
    rng = np.random.default_rng(7)
    for split, count in (("train", 8), ("val", 4)):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = np.full((size, size, 3), 24, dtype=np.uint8)
            class_id = index % 2
            width = int(rng.integers(16, 27))
            height = int(rng.integers(16, 27))
            x = int(rng.integers(1, size - width - 1))
            y = int(rng.integers(1, size - height - 1))
            color = (20, 20, 235) if class_id == 0 else (20, 235, 20)
            cv2.rectangle(image, (x, y), (x + width, y + height), color, -1)
            if not cv2.imwrite(str(image_dir / f"{index:02d}.jpg"), image):
                raise RuntimeError("could not write detection fixture")
            (label_dir / f"{index:02d}.txt").write_text(
                f"{class_id} {(x + width / 2) / size:.6f} {(y + height / 2) / size:.6f} "
                f"{width / size:.6f} {height / size:.6f}\n"
            )
    dataset = root / "dataset.toml"
    dataset.write_text(
        f'path = "{root}"\ntrain = "images/train"\nval = "images/val"\n\n'
        '[names]\n0 = "red-box"\n1 = "green-box"\n'
    )
    return dataset


def _video(path: Path, direction: int, size: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (size, size))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG video writer is unavailable")
    for frame_index in range(6):
        image = np.full((size, size, 3), 24, dtype=np.uint8)
        x = 3 + frame_index * 4 if direction > 0 else size - 11 - frame_index * 4
        cv2.rectangle(image, (x, 14), (x + 8, 25), (40, 180, 245), -1)
        writer.write(image)
    writer.release()


def video_data(root: Path) -> Path:
    for split in ("train", "val"):
        for name, direction in (("move-left", -1), ("move-right", 1)):
            _video(root / split / name / f"{split}.avi", direction)
    return root


def vlm_data(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, answer in enumerate(("red square", "green circle")):
        image = np.full((32, 32, 3), 30, dtype=np.uint8)
        if index == 0:
            cv2.rectangle(image, (8, 8), (24, 24), (20, 20, 230), -1)
        else:
            cv2.circle(image, (16, 16), 9, (20, 230, 20), -1)
        path = root / f"{index}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError("could not write VLM fixture")
        records.append({"image": path.name, "prompt": "Describe the shape.", "answer": answer})
    manifest = root / "train.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    return manifest


def run_detection(root: Path, device: str) -> dict[str, Any]:
    from trainer import Model

    dataset = detection_data(root / "data")
    run_root = root / "run"
    model = Model("trainer-n")
    metrics = model.train(
        str(dataset), epochs=1, imgsz=64, batch=2, workers=0, device=device,
        project=str(run_root), name="train", mosaic=0.25, warmup_epochs=0.0,
        patience=10, val_period=1, amp=device.startswith("cuda"), deterministic=True,
    )
    checkpoint = run_root / "train" / "weights" / "best.pt"
    if not checkpoint.is_file():
        fallback = run_root / "train" / "best.pt"
        checkpoint = fallback if fallback.is_file() else checkpoint
    if not checkpoint.is_file():
        raise RuntimeError("detection training did not create best.pt")
    return {"checkpoint": str(checkpoint), "metrics": metrics, "dataset": str(dataset)}


def run_classification(root: Path, device: str) -> dict[str, Any]:
    from trainer.cli import main

    dataset = image_classification_data(root / "data")
    output = root / "run"
    status = main([
        "train", "task=classification", "arch=convnext-tiny", f"data={dataset}", f"output={output}",
        "epochs=1", "batch=2", "imgsz=48", "width=4", "workers=0", f"device={device}",
        "mosaic=0.25", "mixup=0.25",
    ])
    checkpoint = output / "weights" / "best.pt"
    if status != 0 or not checkpoint.is_file():
        raise RuntimeError(f"classification training failed with status {status}")
    return {"checkpoint": str(checkpoint), "metrics": str(output / "metrics.json"), "dataset": str(dataset)}


def run_video(root: Path, device: str) -> dict[str, Any]:
    from trainer.cli import main

    dataset = video_data(root / "data")
    output = root / "run"
    status = main([
        "train", "task=video", "arch=r2plus1d-tiny", f"data={dataset}", f"output={output}",
        "epochs=1", "batch=2", "imgsz=40", "clip_len=3", "width=4", "hidden=8",
        "workers=0", f"device={device}", "mixup=0.25",
    ])
    checkpoint = output / "best.pt"
    if status != 0 or not checkpoint.is_file():
        raise RuntimeError(f"video training failed with status {status}")
    return {"checkpoint": str(checkpoint), "metrics": str(output / "metrics.json"), "dataset": str(dataset)}


def run_vlm(root: Path, device: str) -> dict[str, Any]:
    from automation.vlm_tune import load_records
    from trainer.smoke import smoke_architecture

    dataset = vlm_data(root / "data")
    records = load_records(dataset)
    result = smoke_architecture("vlm", "tiny-vlm-lora", device)
    output = root / "run"
    output.mkdir(parents=True, exist_ok=True)
    result.update({"dataset": str(dataset), "records": len(records), "mode": "offline-tiny-lora"})
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"metrics": str(output / "metrics.json"), **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "vm_smoke_matrix")
    parser.add_argument("--overwrite", action="store_true", help="replace family folders under an existing output")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {output}; choose another path or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    runners: list[tuple[str, Callable[[Path, str], dict[str, Any]]]] = [
        ("detection", run_detection),
        ("classification", run_classification),
        ("video-classification", run_video),
        ("vlm", run_vlm),
    ]
    summary: dict[str, Any] = {
        "format": "trainer-vm-smoke-matrix-v1",
        "device": args.device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "started_at": time.time(),
        "families": {},
    }
    for name, runner in runners:
        family_root = output / name
        if family_root.exists() and args.overwrite:
            shutil.rmtree(family_root)
        started = time.monotonic()
        print(f"\n== {name} ({args.device}) ==", flush=True)
        try:
            detail = runner(family_root, args.device)
            summary["families"][name] = {
                "status": "passed", "seconds": round(time.monotonic() - started, 3), **detail,
            }
        except Exception as error:  # keep running so the matrix reports every family
            traceback.print_exc()
            summary["families"][name] = {
                "status": "failed", "seconds": round(time.monotonic() - started, 3),
                "error": f"{type(error).__name__}: {error}",
            }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    summary["ended_at"] = time.time()
    summary["ok"] = all(value["status"] == "passed" for value in summary["families"].values())
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
