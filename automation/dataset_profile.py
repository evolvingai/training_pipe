#!/usr/bin/env python3
"""Build a bounded, family-neutral dataset profile for the Trainer UI."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def profile_base(family: str, source: Path, format_name: str) -> dict:
    return {
        "schema": "trainer-dataset-profile-v1", "family": family, "format": format_name,
        "name": source.stem if source.is_file() else source.name, "source": str(source),
        "generatedAt": int(time.time() * 1000), "summary": {}, "splits": [], "classes": [],
        "boxSizes": {}, "extensions": {}, "text": {},
    }


def split_name(path: Path) -> str:
    value = path.stem
    return value.removeprefix("instances_").removeprefix("person_keypoints_").removeprefix("captions_")


def coco_annotation_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    candidates = [*source.glob("instances_*.json"), *source.glob("annotations/instances_*.json")]
    if not candidates:
        candidates = [item for item in source.glob("*.json") if item.is_file()]
    unique = {item.resolve(): item for item in candidates}
    if not unique:
        raise FileNotFoundError(f"no COCO annotation JSON found under {source}")
    return [unique[key] for key in sorted(unique)]


def profile_coco(source: Path) -> dict:
    result = profile_base("detection", source, "coco")
    class_names: dict[int, str] = {}
    class_counts: Counter = Counter()
    class_splits: dict[int, Counter] = defaultdict(Counter)
    total_images = total_annotations = total_empty = invalid = crowd = 0
    box_sizes: Counter = Counter()
    splits = []
    for annotation_file in coco_annotation_files(source):
        payload = json.loads(annotation_file.read_text(encoding="utf8"))
        images = {int(item["id"]): item for item in payload.get("images", []) if "id" in item}
        categories = {int(item["id"]): str(item.get("name", item["id"])) for item in payload.get("categories", []) if "id" in item}
        class_names.update(categories)
        annotated = set()
        split_classes: set[int] = set()
        split_invalid = split_crowd = 0
        split = split_name(annotation_file)
        annotations = payload.get("annotations", [])
        for annotation in annotations:
            image_id = int(annotation.get("image_id", -1))
            category_id = int(annotation.get("category_id", -1))
            bbox = annotation.get("bbox") or []
            if image_id not in images or category_id not in categories or len(bbox) < 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                invalid += 1; split_invalid += 1
                continue
            annotated.add(image_id); split_classes.add(category_id)
            class_counts[category_id] += 1; class_splits[category_id][split] += 1
            area = float(annotation.get("area") or (float(bbox[2]) * float(bbox[3])))
            box_sizes["small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"] += 1
            if annotation.get("iscrowd"):
                crowd += 1; split_crowd += 1
        empty = max(0, len(images) - len(annotated))
        valid_annotations = len(annotations) - split_invalid
        total_images += len(images); total_annotations += valid_annotations; total_empty += empty
        splits.append({"name": split, "samples": len(images), "images": len(images),
                       "annotations": valid_annotations, "labeled": len(annotated), "unlabeled": empty,
                       "classes": len(split_classes), "invalid": split_invalid, "crowd": split_crowd})
    result["splits"] = splits
    result["classes"] = [{"id": category_id, "name": class_names.get(category_id, str(category_id)),
                           "total": class_counts[category_id], "splits": dict(class_splits[category_id])}
                          for category_id in sorted(class_names)]
    result["boxSizes"] = {key: box_sizes[key] for key in ("small", "medium", "large")}
    result["summary"] = {"samples": total_images, "images": total_images, "annotations": total_annotations,
                         "classes": len(class_names), "labeled": total_images - total_empty,
                         "unlabeled": total_empty, "invalid": invalid, "crowd": crowd}
    return result


def profile_coco_multilabel(source: Path) -> dict:
    """Profile image-level category presence derived from COCO instances."""
    result = profile_base("classification", source, "coco-multilabel")
    class_names: dict[int, str] = {}
    class_counts: Counter = Counter()
    class_splits: dict[int, Counter] = defaultdict(Counter)
    total_images = total_labels = total_empty = invalid = crowd = 0
    splits = []
    for annotation_file in coco_annotation_files(source):
        payload = json.loads(annotation_file.read_text(encoding="utf8"))
        images = {int(item["id"]): item for item in payload.get("images", []) if "id" in item}
        categories = {int(item["id"]): str(item.get("name", item["id"]))
                      for item in payload.get("categories", []) if "id" in item}
        class_names.update(categories)
        positives: dict[int, set[int]] = defaultdict(set)
        split_invalid = split_crowd = 0
        split = split_name(annotation_file)
        for annotation in payload.get("annotations", []):
            image_id = int(annotation.get("image_id", -1))
            category_id = int(annotation.get("category_id", -1))
            if image_id not in images or category_id not in categories:
                invalid += 1; split_invalid += 1
                continue
            positives[image_id].add(category_id)
            if annotation.get("iscrowd"):
                crowd += 1; split_crowd += 1
        split_classes: set[int] = set()
        for image_labels in positives.values():
            for category_id in image_labels:
                class_counts[category_id] += 1
                class_splits[category_id][split] += 1
                split_classes.add(category_id)
        positive_count = sum(len(labels) for labels in positives.values())
        labeled = sum(bool(labels) for labels in positives.values())
        empty = max(0, len(images) - labeled)
        total_images += len(images); total_labels += positive_count; total_empty += empty
        splits.append({
            "name": split, "samples": len(images), "images": len(images),
            "annotations": positive_count, "labeled": labeled, "unlabeled": empty,
            "classes": len(split_classes), "invalid": split_invalid, "crowd": split_crowd,
        })
    result["splits"] = splits
    result["classes"] = [
        {"id": category_id, "name": class_names.get(category_id, str(category_id)),
         "total": class_counts[category_id], "splits": dict(class_splits[category_id])}
        for category_id in sorted(class_names)
    ]
    result["summary"] = {
        "samples": total_images, "images": total_images, "annotations": total_labels,
        "classes": len(class_names), "labeled": total_images - total_empty,
        "unlabeled": total_empty, "invalid": invalid, "crowd": crowd,
    }
    return result


def source_files(value, root: Path, extensions: set[str]) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    output: list[Path] = []
    for item in values:
        candidate = Path(str(item))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_dir():
            output.extend(path for path in candidate.rglob("*") if path.suffix.lower() in extensions)
        elif candidate.is_file() and candidate.suffix.lower() == ".txt":
            for line in candidate.read_text(encoding="utf8").splitlines():
                path = Path(line.strip())
                if not path.is_absolute():
                    path = candidate.parent / path
                if path.suffix.lower() in extensions:
                    output.append(path)
        elif candidate.suffix.lower() in extensions:
            output.append(candidate)
    return sorted(set(path.resolve() for path in output))


def yolo_label_path(image: Path) -> Path:
    parts = list(image.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def profile_yolo(source: Path) -> dict:
    config = tomllib.loads(source.read_text(encoding="utf8"))
    root = Path(str(config.get("path") or source.parent))
    if not root.is_absolute():
        root = (source.parent / root).resolve()
    raw_names = config.get("names") or {}
    names = {int(key): str(value) for key, value in raw_names.items()} if isinstance(raw_names, dict) else {
        index: str(value) for index, value in enumerate(raw_names)}
    result = profile_base("detection", source, "yolo")
    counts: Counter = Counter(); split_counts: dict[int, Counter] = defaultdict(Counter)
    extensions: Counter = Counter(); sizes: Counter = Counter()
    splits = []; total_images = total_annotations = total_empty = invalid = 0
    for split in ("train", "val", "test"):
        if split not in config:
            continue
        images = source_files(config[split], root, IMAGE_EXTENSIONS)
        annotated_images = annotations = split_invalid = 0
        split_classes: set[int] = set()
        for image in images:
            extensions[image.suffix.lower().lstrip(".")] += 1
            label = yolo_label_path(image)
            valid_rows = 0
            if label.is_file():
                for line in label.read_text(encoding="utf8").splitlines():
                    values = line.split()
                    try:
                        class_id = int(float(values[0])); width = float(values[3]); height = float(values[4])
                        if len(values) < 5 or width <= 0 or height <= 0 or width > 1 or height > 1:
                            raise ValueError
                    except (ValueError, IndexError):
                        invalid += 1; split_invalid += 1
                        continue
                    valid_rows += 1; annotations += 1; counts[class_id] += 1; split_counts[class_id][split] += 1
                    split_classes.add(class_id)
                    area = width * height
                    sizes["small" if area < .0025 else "medium" if area < .0225 else "large"] += 1
            if valid_rows:
                annotated_images += 1
        empty = len(images) - annotated_images
        total_images += len(images); total_annotations += annotations; total_empty += empty
        splits.append({"name": split, "samples": len(images), "images": len(images), "annotations": annotations,
                       "labeled": annotated_images, "unlabeled": empty, "classes": len(split_classes), "invalid": split_invalid})
    all_ids = sorted(set(names) | set(counts))
    result["splits"] = splits
    result["classes"] = [{"id": class_id, "name": names.get(class_id, str(class_id)), "total": counts[class_id],
                           "splits": dict(split_counts[class_id])} for class_id in all_ids]
    result["boxSizes"] = {key: sizes[key] for key in ("small", "medium", "large")}
    result["extensions"] = dict(extensions)
    result["summary"] = {"samples": total_images, "images": total_images, "annotations": total_annotations,
                         "classes": len(all_ids), "labeled": total_images - total_empty,
                         "unlabeled": total_empty, "invalid": invalid, "crowd": 0}
    return result


def class_folder_splits(source: Path) -> list[tuple[str, Path]]:
    explicit = [(name, source / name) for name in ("train", "val", "test") if (source / name).is_dir()]
    return explicit or [("all", source)]


def profile_class_folders(source: Path, family: str) -> dict:
    extensions_set = IMAGE_EXTENSIONS if family == "classification" else VIDEO_EXTENSIONS
    format_name = "class-folders" if family == "classification" else "video-folders"
    result = profile_base(family, source, format_name)
    counts: Counter = Counter(); class_splits: dict[str, Counter] = defaultdict(Counter); extensions: Counter = Counter()
    splits = []; total = 0
    for split, folder in class_folder_splits(source):
        split_total = 0; split_classes = 0
        for class_dir in sorted(item for item in folder.iterdir() if item.is_dir()):
            files = [item for item in class_dir.rglob("*") if item.suffix.lower() in extensions_set]
            if files:
                split_classes += 1; split_total += len(files); counts[class_dir.name] += len(files)
                class_splits[class_dir.name][split] += len(files)
                for item in files:
                    extensions[item.suffix.lower().lstrip(".")] += 1
        total += split_total
        splits.append({"name": split, "samples": split_total, "images": split_total if family == "classification" else 0,
                       "annotations": split_total, "labeled": split_total, "unlabeled": 0, "classes": split_classes, "invalid": 0})
    result["splits"] = splits
    result["classes"] = [{"id": index, "name": name, "total": count, "splits": dict(class_splits[name])}
                          for index, (name, count) in enumerate(sorted(counts.items()))]
    result["extensions"] = dict(extensions)
    result["summary"] = {"samples": total, "images": total if family == "classification" else 0,
                         "annotations": total, "classes": len(counts), "labeled": total,
                         "unlabeled": 0, "invalid": 0, "crowd": 0}
    return result


def profile_vlm(source: Path) -> dict:
    result = profile_base("vlm", source, "vlm-jsonl")
    samples = missing = invalid = prompt_chars = answer_chars = 0
    for line in source.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line); image = str(value.get("image") or "")
            prompt = str(value.get("prompt") or ""); answer = str(value.get("answer") or "")
            if not image or not prompt or not answer:
                invalid += 1; continue
            image_path = Path(image) if Path(image).is_absolute() else source.parent / image
            missing += int(not image_path.is_file()); samples += 1
            prompt_chars += len(prompt); answer_chars += len(answer)
        except (json.JSONDecodeError, AttributeError):
            invalid += 1
    result["splits"] = [{"name": "all", "samples": samples, "images": samples, "annotations": samples,
                         "labeled": samples - missing, "unlabeled": missing, "classes": 0, "invalid": invalid}]
    result["summary"] = {"samples": samples, "images": samples, "annotations": samples, "classes": 0,
                         "labeled": samples - missing, "unlabeled": missing, "invalid": invalid, "crowd": 0}
    result["text"] = {"averagePromptCharacters": prompt_chars / max(samples, 1),
                      "averageAnswerCharacters": answer_chars / max(samples, 1)}
    return result


def build_profile(family: str, source: Path) -> dict:
    source = source.resolve()
    if family == "detection":
        if source.suffix.lower() == ".toml":
            return profile_yolo(source)
        return profile_coco(source)
    has_coco_annotations = source.is_dir() and bool([
        *source.glob("instances_*.json"), *source.glob("annotations/instances_*.json"),
    ])
    if family == "classification" and (source.suffix.lower() == ".json" or has_coco_annotations):
        return profile_coco_multilabel(source)
    if family in {"classification", "video"}:
        if not source.is_dir():
            raise NotADirectoryError(f"class-folder dataset does not exist: {source}")
        return profile_class_folders(source, family)
    if family == "vlm":
        return profile_vlm(source)
    raise ValueError(f"unsupported family: {family}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Profile a COCO or custom Trainer dataset")
    value.add_argument("--family", choices=["detection", "classification", "video", "vlm"], required=True)
    value.add_argument("--data", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    profile = build_profile(args.family, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf8")
    print(f"profiled {profile['format']} dataset: {profile['summary'].get('samples', 0)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
