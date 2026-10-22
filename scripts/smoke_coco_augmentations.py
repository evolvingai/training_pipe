#!/usr/bin/env python3
"""End-to-end COCO augmentation smoke test for Trainer tasks.

The script downloads eight small public COCO validation samples through the
Hugging Face dataset viewer, builds a two-class COCO detection subset, derives
classification crops, and synthesizes short left/right action clips.  One
external augmentation TOML is then used by:

* the existing Trainer detection ``Model`` facade;
* the Trainer ConvNeXt image-classification lifecycle; and
* the Trainer R(2+1)D video/action-classification lifecycle.

Outputs include machine-readable checks plus contact sheets that place source
labels, transformed labels, and predictions side by side.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from automation import video_classifier
from trainer.augmentation import build_pipeline, load_augmentation_config
from trainer.classification import engine as classification
from trainer.classification.data import ImageDataset, Sample as ImageSample
from trainer.detection import Model
from trainer.detection.data import CocoDetectionDataset
from trainer.detection.utils.plotting import draw_boxes

DATASET_ROWS_URL = (
    "https://datasets-server.huggingface.co/first-rows?"
    "dataset=finedet%2Fcoco2017&config=default&split=validation"
)
TARGET_ROWS = {"banana": (31, 39, 57, 90), "bowl": (23, 37, 55, 85)}
COLORS = {"ok": "#31c48d", "warn": "#f59e0b", "bad": "#f05252"}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Trainer-COCO-smoke/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item: object) -> object:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"cannot serialize {type(item).__name__}")

    path.write_text(json.dumps(value, indent=2, default=convert) + "\n", encoding="utf8")


def _category_names(metadata: dict[str, Any]) -> list[str]:
    objects = next(feature for feature in metadata["features"] if feature["name"] == "objects")
    return list(objects["type"]["category"]["feature"]["names"])


def _selected_rows(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    names = _category_names(metadata)
    wanted = {row for rows in TARGET_ROWS.values() for row in rows}
    output = []
    for wrapped in metadata["rows"]:
        row_index = int(wrapped["row_idx"])
        if row_index not in wanted:
            continue
        row = wrapped["row"]
        objects = row["objects"]
        candidates = []
        for bbox, category in zip(objects["bbox"], objects["category"], strict=True):
            category_name = names[int(category)]
            if row_index in TARGET_ROWS.get(category_name, ()):
                candidates.append((bbox, category_name))
        if not candidates:
            raise RuntimeError(f"COCO validation row {row_index} no longer contains its expected class")
        output.append({
            "row": row_index,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "url": row["image"]["src"],
            "objects": candidates,
        })
    if len(output) != len(wanted):
        found = {item["row"] for item in output}
        raise RuntimeError(f"expected COCO rows {sorted(wanted)}, got {sorted(found)}")
    return sorted(output, key=lambda item: item["row"])


def prepare_data(output: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    source = output / "source"
    metadata_file = metadata_path or source / "coco_validation_first_rows.json"
    if not metadata_file.exists():
        print(f"downloading COCO row metadata -> {metadata_file}")
        _download(DATASET_ROWS_URL, metadata_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf8"))
    rows = _selected_rows(metadata)

    images_dir = source / "images"
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(TARGET_ROWS)]
    category_id = {item["name"]: int(item["id"]) for item in categories}
    coco_images, annotations = [], []
    annotation_id = 1
    manifest_rows = []
    for item in rows:
        image_path = images_dir / f"coco_validation_row_{item['row']:05d}.jpg"
        if not image_path.exists():
            print(f"downloading COCO validation row {item['row']} -> {image_path.name}")
            _download(item["url"], image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"downloaded image is unreadable: {image_path}")
        height, width = image.shape[:2]
        if (width, height) != (item["width"], item["height"]):
            raise RuntimeError(
                f"COCO row {item['row']} dimensions changed: expected {(item['width'], item['height'])}, "
                f"got {(width, height)}"
            )
        image_id = item["row"] + 1
        coco_images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        row_labels = []
        for bbox, name in item["objects"]:
            x, y, box_width, box_height = (float(value) for value in bbox)
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id[name],
                "bbox": [x, y, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            })
            row_labels.append({"class": name, "bbox": [x, y, box_width, box_height]})
            annotation_id += 1
        manifest_rows.append({
            "row": item["row"], "image": str(image_path), "width": width, "height": height,
            "labels": row_labels,
        })

    annotations_path = source / "instances_smoke.json"
    _save_json(annotations_path, {
        "info": {"description": "Trainer public COCO augmentation smoke subset"},
        "licenses": [], "images": coco_images, "annotations": annotations, "categories": categories,
    })
    _save_json(source / "manifest.json", {
        "dataset": "COCO 2017 validation via finedet/coco2017 public mirror",
        "metadata_url": DATASET_ROWS_URL,
        "classes": list(TARGET_ROWS),
        "rows": manifest_rows,
    })

    detector_toml = source / "detection.toml"
    detector_toml.write_text(
        f'path = "{source}"\n\n'
        '[train]\nformat = "coco"\nimages = "images"\nannotations = "instances_smoke.json"\n\n'
        '[val]\nformat = "coco"\nimages = "images"\nannotations = "instances_smoke.json"\n',
        encoding="utf8",
    )
    augmentation_toml = output / "augmentation.toml"
    augmentation_toml.write_text(
        """[augmentation]
mosaic = 0.35
mixup = 0.15
mixup_alpha = 1.2
copy_paste = 0.10
degrees = 10.0
translate = 0.08
scale = 0.20
fliplr = 0.50
hsv_h = 0.01
hsv_s = 0.25
hsv_v = 0.20
brightness_contrast = 0.40
brightness = 0.15
contrast = 0.15
blur = 0.10
noise = 0.10
jpeg = 0.10
min_box_visibility = 0.15
""",
        encoding="utf8",
    )
    classification_root = _prepare_classification(output, manifest_rows)
    video_root = _prepare_videos(output, manifest_rows)
    return {
        "rows": manifest_rows,
        "images_dir": images_dir,
        "annotations": annotations_path,
        "detector_toml": detector_toml,
        "augmentation_toml": augmentation_toml,
        "classification_root": classification_root,
        "video_root": video_root,
    }


def _object_crop(row: dict[str, Any], margin: float = 0.15) -> np.ndarray:
    image = cv2.imread(row["image"])
    if image is None:
        raise RuntimeError(f"could not read {row['image']}")
    label = max(row["labels"], key=lambda value: value["bbox"][2] * value["bbox"][3])
    x, y, width, height = label["bbox"]
    pad_x, pad_y = width * margin, height * margin
    x1, y1 = max(0, int(x - pad_x)), max(0, int(y - pad_y))
    x2 = min(image.shape[1], int(np.ceil(x + width + pad_x)))
    y2 = min(image.shape[0], int(np.ceil(y + height + pad_y)))
    return image[y1:y2, x1:x2].copy()


def _prepare_classification(output: Path, rows: list[dict[str, Any]]) -> Path:
    root = output / "datasets" / "classification"
    for row in rows:
        class_name = row["labels"][0]["class"]
        crop = _object_crop(row)
        for split in ("train", "val"):
            destination = root / split / class_name / f"row_{row['row']:05d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), crop)
    return root


def _motion_frames(image: np.ndarray, direction: str, frame_count: int = 8, size: int = 80) -> list[np.ndarray]:
    height, width = image.shape[:2]
    scale = max(size / max(height, 1), (size * 1.45) / max(width, 1))
    resized = cv2.resize(image, (max(size + 4, round(width * scale)), max(size, round(height * scale))))
    if resized.shape[0] < size:
        resized = cv2.copyMakeBorder(resized, 0, size - resized.shape[0], 0, 0, cv2.BORDER_REFLECT_101)
    max_x = max(0, resized.shape[1] - size)
    positions = np.linspace(0, max_x, frame_count).round().astype(int)
    if direction == "pan_left":
        positions = positions[::-1]
    y = max(0, (resized.shape[0] - size) // 2)
    return [resized[y:y + size, int(x):int(x) + size].copy() for x in positions]


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 8.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not create video: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def _prepare_videos(output: Path, rows: list[dict[str, Any]]) -> Path:
    root = output / "datasets" / "video"
    for row in rows:
        crop = _object_crop(row, margin=0.35)
        for direction in ("pan_left", "pan_right"):
            frames = _motion_frames(crop, direction)
            for split in ("train", "val"):
                _write_video(root / split / direction / f"row_{row['row']:05d}.avi", frames)
    return root


def _panel(image: np.ndarray, title: str, subtitle: str = "", size: int = 256) -> np.ndarray:
    canvas = np.full((size + 54, size, 3), 245, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, size), (size, size + 54), (24, 31, 45), -1)
    cv2.putText(canvas, title[:35], (8, size + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle[:43], (8, size + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (190, 201, 220), 1, cv2.LINE_AA)
    return canvas


def _grid(rows: list[list[np.ndarray]], destination: Path) -> Path:
    row_images = [np.concatenate(row, axis=1) for row in rows]
    image = np.concatenate(row_images, axis=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image)
    return destination


def _xywh_to_xyxy(labels: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    boxes, classes = [], []
    names = list(TARGET_ROWS)
    for label in labels:
        x, y, width, height = label["bbox"]
        boxes.append([x, y, x + width, y + height])
        classes.append(names.index(label["class"]))
    return np.asarray(boxes, np.float32), np.asarray(classes, np.int64)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return intersection / max(left_area + right_area - intersection, 1e-9)


def run_detection(paths: dict[str, Any], output: Path, epochs: int, reuse: bool = False) -> tuple[dict[str, Any], Path]:
    run_root = output / "runs"
    checkpoint = run_root / "detection" / "weights" / "best.pt"
    if reuse and checkpoint.exists():
        model = Model(checkpoint, device="cpu")
        previous = json.loads((output / "summary.json").read_text()) if (output / "summary.json").exists() else {}
        metrics = previous.get("detection", {}).get("train_metrics", {})
    else:
        model = Model("trainer-n", device="cpu")
        metrics = model.train(
            data=str(paths["detector_toml"]), epochs=epochs, batch=4, imgsz=128, workers=0,
            device="cpu", project=str(run_root), name="detection", exist_ok=True,
            augmentation_config=str(paths["augmentation_toml"]), amp=False, channels_last=False,
            warmup_epochs=0.0, patience=epochs + 1, early_stop_min_epochs=epochs + 1,
        )
    augmentation = load_augmentation_config(paths["augmentation_toml"])
    augmentation["imgsz"] = 128
    dataset = CocoDetectionDataset(
        paths["images_dir"], paths["annotations"], imgsz=128,
        transforms=build_pipeline(augmentation, train=True, seed=17), cache="none",
    )
    preview_paths, augmented_samples = [], []
    preview_indices = (0, 1)  # one bowl and one banana in the row-sorted subset
    for index in preview_indices:
        item = dataset[index]
        image = item["img"].numpy().transpose(1, 2, 0)
        h, w = image.shape[:2]
        boxes = item["bboxes"].numpy()
        xyxy = np.column_stack([
            (boxes[:, 0] - boxes[:, 2] / 2) * w,
            (boxes[:, 1] - boxes[:, 3] / 2) * h,
            (boxes[:, 0] + boxes[:, 2] / 2) * w,
            (boxes[:, 1] + boxes[:, 3] / 2) * h,
        ]).astype(np.float32) if len(boxes) else np.zeros((0, 4), np.float32)
        preview_path = output / "previews" / f"detection_augmented_{index}.jpg"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), image)
        preview_paths.append(preview_path)
        augmented_samples.append((image, xyxy, item["cls"].numpy().astype(int)))

    predictions = model.predict([str(path) for path in preview_paths], conf=0.001, batch_size=2)
    names = {0: "banana", 1: "bowl"}
    visual_rows, matched_ious = [], []
    for source_row, (aug_image, aug_boxes, aug_classes), prediction in zip(
        (paths["rows"][index] for index in preview_indices), augmented_samples, predictions, strict=True,
    ):
        source = cv2.imread(source_row["image"])
        source_boxes, source_classes = _xywh_to_xyxy(source_row["labels"])
        source_viz = draw_boxes(source, source_boxes, source_classes, names)
        augmented_viz = draw_boxes(aug_image, aug_boxes, aug_classes, names)
        audit_indices, sample_ious = [], []
        for gt_box, gt_class in zip(aug_boxes, aug_classes, strict=True):
            candidates = [
                (_iou(gt_box, predicted_box), candidate_index)
                for candidate_index, (predicted_box, predicted_class) in enumerate(zip(
                    prediction.boxes, prediction.cls, strict=True
                )) if int(predicted_class) == int(gt_class)
            ]
            best_iou, best_index = max(candidates, default=(0.0, None))
            matched_ious.append(best_iou)
            sample_ious.append(best_iou)
            if best_index is not None:
                audit_indices.append(best_index)
        audit = np.asarray(sorted(set(audit_indices)), dtype=np.int64)
        prediction_viz = draw_boxes(
            aug_image, prediction.boxes[audit], prediction.cls[audit], names, prediction.conf[audit],
        )
        visual_rows.append([
            _panel(source_viz, "COCO source + GT", source_row["labels"][0]["class"]),
            _panel(augmented_viz, "Augmented + moved GT", f"{len(aug_boxes)} transformed box(es)"),
            _panel(prediction_viz, "Best class-matched prediction",
                   f"audit IoU {np.mean(sample_ious):.2f}; {len(prediction)} raw proposals"),
        ])
    sheet = _grid(visual_rows, output / "visuals" / "detection.jpg")
    result = {
        "architecture": "trainer-n",
        "framework": "trainer.detection.Model",
        "trained_epochs": max(0, sum(1 for _ in (run_root / "detection" / "logs" / "results.csv").open()) - 1),
        "best_checkpoint_epoch": int(model.ckpt.get("epoch", -1)) + 1 if model.ckpt else None,
        "train_metrics": metrics,
        "transformed_boxes_valid": all(
            len(boxes) > 0 and np.all(boxes >= 0) and np.all(boxes[:, [0, 2]] <= image.shape[1])
            and np.all(boxes[:, [1, 3]] <= image.shape[0])
            for image, boxes, _ in augmented_samples
        ),
        "prediction_count": sum(len(item) for item in predictions),
        "mean_best_class_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "checkpoint": str(checkpoint),
    }
    return result, sheet


def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    rgb = np.clip((tensor.detach().cpu().numpy().transpose(1, 2, 0) * 0.25 + 0.5) * 255, 0, 255)
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def _classify_tensor(model: torch.nn.Module, tensor: torch.Tensor, names: list[str]) -> tuple[str, float]:
    with torch.inference_mode():
        probabilities = model(tensor.unsqueeze(0)).softmax(1)[0]
    confidence, index = probabilities.max(0)
    return names[int(index)], float(confidence)


def run_classification(paths: dict[str, Any], output: Path, epochs: int, reuse: bool = False) -> tuple[dict[str, Any], Path]:
    run_dir = output / "runs" / "classification"
    checkpoint = run_dir / "weights" / "best.pt"
    if not (reuse and checkpoint.exists()):
        classification.main([
            "train", "--data", str(paths["classification_root"]), "--output", str(run_dir),
            "--arch", "convnext-tiny", "--epochs", str(epochs), "--batch", "4", "--imgsz", "96",
            "--width", "12", "--dropout", "0.0", "--optimizer", "adamw", "--lr", "0.003",
            "--device", "cpu", "--workers", "0", "--no-amp", "--no-channels-last",
            "--patience", str(epochs + 1), "--min-epochs", str(epochs + 1),
            "--augmentation-config", str(paths["augmentation_toml"]),
        ])
    model, payload = classification.load_checkpoint(checkpoint, torch.device("cpu"))
    model.eval()
    names = list(payload["classes"])
    samples = []
    for name in names:
        image_path = sorted((paths["classification_root"] / "val" / name).glob("*.jpg"))[0]
        samples.append(ImageSample(image_path, names.index(name)))
    augmentation = load_augmentation_config(paths["augmentation_toml"])
    dataset = ImageDataset(samples, 96, augment=True, num_classes=len(names), augmentation=augmentation, seed=0)
    eval_dataset = ImageDataset(samples, 96, augment=False, num_classes=len(names), seed=0)
    visual_rows, correct, stable = [], 0, 0
    for index, sample in enumerate(samples):
        original_tensor, actual_index, _ = eval_dataset[index]
        augmented_tensor, target, _ = dataset[index]
        original_prediction, original_conf = _classify_tensor(model, original_tensor, names)
        augmented_prediction, augmented_conf = _classify_tensor(model, augmented_tensor, names)
        actual = names[int(actual_index)]
        target_np = target.numpy()
        target_text = ", ".join(f"{name} {target_np[i]:.2f}" for i, name in enumerate(names) if target_np[i] > 0.01)
        dominant_target = names[int(target.argmax())]
        correct += int(original_prediction == actual)
        stable += int(augmented_prediction == dominant_target)
        source = cv2.imread(str(sample.path))
        visual_rows.append([
            _panel(source, "COCO object crop", f"GT: {actual}"),
            _panel(_tensor_to_bgr(augmented_tensor), "Shared augmentation", f"target: {target_text}"),
            _panel(_tensor_to_bgr(augmented_tensor), "ConvNeXt prediction",
                   f"{augmented_prediction} {augmented_conf:.2f} (source {original_conf:.2f})"),
        ])
    sheet = _grid(visual_rows, output / "visuals" / "classification.jpg")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf8"))
    return {
        "architecture": "convnext-tiny",
        "trained_epochs": len(metrics["history"]),
        "best_checkpoint_epoch": int(payload.get("epoch", 0)),
        "source_label_matches": f"{correct}/{len(samples)}",
        "augmented_target_matches": f"{stable}/{len(samples)}",
        "validation_accuracy": metrics["final"]["accuracy"],
        "best_validation_accuracy": metrics["best_accuracy"],
        "checkpoint": str(checkpoint),
    }, sheet


def _video_prediction(model: torch.nn.Module, clip: torch.Tensor, names: list[str]) -> tuple[str, float]:
    with torch.inference_mode():
        probabilities = model(clip.unsqueeze(0)).softmax(1)[0]
    confidence, index = probabilities.max(0)
    return names[int(index)], float(confidence)


def run_video(paths: dict[str, Any], output: Path, epochs: int, reuse: bool = False) -> tuple[dict[str, Any], Path]:
    run_dir = output / "runs" / "video"
    checkpoint = run_dir / "best.pt"
    if not (reuse and checkpoint.exists()):
        video_classifier.main([
            "train", "--data", str(paths["video_root"]), "--output", str(run_dir),
            "--arch", "r2plus1d-tiny", "--epochs", str(epochs), "--batch", "4", "--imgsz", "64",
            "--clip-len", "8", "--width", "6", "--dropout", "0.0", "--lr", "0.003",
            "--device", "cpu", "--workers", "0", "--augmentation-config", str(paths["augmentation_toml"]),
        ])
    model, payload = video_classifier.load_checkpoint(checkpoint, torch.device("cpu"))
    model.eval()
    names = list(payload["classes"])
    augmentation = load_augmentation_config(paths["augmentation_toml"])
    augmentation["imgsz"] = 64
    transforms = build_pipeline(augmentation, train=True, seed=29, allow_multisample=False)
    visual_rows, matches = [], 0
    for name in names:
        video_path = sorted((paths["video_root"] / "val" / name).glob("*.avi"))[0]
        clip = video_classifier.read_clip(video_path, 8, 64, False)
        augmented = video_classifier.read_clip(video_path, 8, 64, False, transforms)
        prediction, confidence = _video_prediction(model, augmented, names)
        matches += int(prediction == name)
        frame_indices = (0, 3, 7)
        panels = []
        for position, frame_index in enumerate(frame_indices):
            title = ("Augmented clip: first", "Augmented clip: middle", "Augmented clip: last")[position]
            subtitle = f"GT {name}; pred {prediction} {confidence:.2f}"
            panels.append(_panel(_tensor_to_bgr(augmented[frame_index]), title, subtitle))
        visual_rows.append(panels)

    reference_path = sorted((paths["video_root"] / "val" / names[0]).glob("*.avi"))[0]
    capture = cv2.VideoCapture(str(reference_path))
    ok, reference = capture.read(); capture.release()
    if not ok:
        raise RuntimeError(f"could not read video consistency reference: {reference_path}")
    repeated = [{"image": reference.copy(), "boxes": np.zeros((0, 4), np.float32),
                 "cls": np.zeros((0,), np.float32)} for _ in range(4)]
    replay = build_pipeline(augmentation, train=True, seed=37, allow_multisample=False).apply_many(repeated)
    temporal_consistency = all(np.array_equal(replay[0]["image"], sample["image"]) for sample in replay[1:])
    sheet = _grid(visual_rows, output / "visuals" / "video.jpg")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf8"))
    return {
        "architecture": "r2plus1d-tiny",
        "trained_epochs": len(metrics["history"]),
        "best_checkpoint_epoch": int(payload.get("epoch", 0)),
        "augmented_label_matches": f"{matches}/{len(names)}",
        "temporal_transform_replay": temporal_consistency,
        "validation_accuracy": metrics["final"]["accuracy"],
        "best_validation_accuracy": metrics["best_accuracy"],
        "checkpoint": str(checkpoint),
        "note": "Action clips are deterministic left/right pans derived from the public COCO object crops.",
    }, sheet


def _data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def write_visualization(path: Path, sheets: dict[str, Path], summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "Detection boxes valid": summary["detection"]["transformed_boxes_valid"],
        "Classification best val": f"{summary['classification']['best_validation_accuracy']:.0%}",
        "Video best val": f"{summary['video']['best_validation_accuracy']:.0%}",
        "Temporal replay": summary["video"]["temporal_transform_replay"],
    }
    badges = "".join(
        f'<div class="metric"><span>{label}</span><strong>{value}</strong></div>' for label, value in metrics.items()
    )
    figures = "".join(
        f'<figure><figcaption>{name}</figcaption><img src="{_data_uri(image)}" alt="{name} smoke test"></figure>'
        for name, image in sheets.items()
    )
    fragment = f"""<style>
#trainer-coco-smoke {{ color: var(--color-text-primary); background: var(--color-background-primary); font: 14px/1.45 var(--font-sans); padding: 18px; }}
#trainer-coco-smoke .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-bottom:18px; }}
#trainer-coco-smoke .metric {{ border:1px solid var(--color-border-secondary); border-radius:10px; padding:10px 12px; background:var(--color-background-secondary); }}
#trainer-coco-smoke .metric span {{ display:block; color:var(--color-text-secondary); font-size:12px; }}
#trainer-coco-smoke .metric strong {{ display:block; font-size:18px; margin-top:2px; }}
#trainer-coco-smoke figure {{ margin:0 0 22px; }}
#trainer-coco-smoke figcaption {{ font-weight:650; font-size:16px; margin:0 0 8px; }}
#trainer-coco-smoke img {{ display:block; width:100%; height:auto; border:1px solid var(--color-border-secondary); border-radius:8px; }}
@media (max-width:720px) {{ #trainer-coco-smoke .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
</style>
<section id="trainer-coco-smoke"><div class="metrics">{badges}</div>{figures}</section>
"""
    path.write_text(fragment, encoding="utf8")
    if path.stat().st_size >= 1_000_000:
        raise RuntimeError(f"visualization exceeds 1 MB: {path.stat().st_size} bytes")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs" / "coco_augmentation_smoke")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Optional cached Hugging Face first-rows JSON (avoids metadata network request).")
    parser.add_argument("--visualization", type=Path, default=None,
                        help="Optional inline-visualization HTML fragment destination.")
    parser.add_argument("--detection-epochs", type=int, default=12)
    parser.add_argument("--classification-epochs", type=int, default=18)
    parser.add_argument("--video-epochs", type=int, default=12)
    parser.add_argument("--reuse-checkpoints", action="store_true",
                        help="Regenerate checks and visuals from existing checkpoints without retraining.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    if args.metadata is not None:
        args.metadata = args.metadata.resolve()
    if args.visualization is not None:
        args.visualization = args.visualization.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = prepare_data(args.output, args.metadata)
    detection, detection_sheet = run_detection(
        paths, args.output, args.detection_epochs, reuse=args.reuse_checkpoints,
    )
    image_classification, classification_sheet = run_classification(
        paths, args.output, args.classification_epochs, reuse=args.reuse_checkpoints,
    )
    video, video_sheet = run_video(paths, args.output, args.video_epochs, reuse=args.reuse_checkpoints)
    summary = {
        "dataset": {
            "source": "COCO 2017 validation via finedet/coco2017 public mirror",
            "samples": len(paths["rows"]), "classes": list(TARGET_ROWS),
            "augmentation_config": str(paths["augmentation_toml"]),
        },
        "detection": detection,
        "classification": image_classification,
        "video": video,
        "visuals": {
            "detection": str(detection_sheet), "classification": str(classification_sheet),
            "video": str(video_sheet),
        },
    }
    _save_json(args.output / "summary.json", summary)
    if args.visualization:
        visual = write_visualization(args.visualization, {
            "Detection — source GT / transformed GT / framework prediction": detection_sheet,
            "Image classification — crop / augmented target / ConvNeXt prediction": classification_sheet,
            "Video action — temporally consistent augmented frames and R(2+1)D prediction": video_sheet,
        }, summary)
        summary["visualization"] = str(visual)
        _save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
