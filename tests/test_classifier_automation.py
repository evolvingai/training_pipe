import json
from pathlib import Path

import cv2
import numpy as np
import torch

from trainer.classification.data import ImageDataset, TinyClassifier, classification_mode, discover_training
from trainer.cli import main


def _dataset(root: Path) -> None:
    for class_name, color in (("dark", 20), ("light", 235)):
        folder = root / class_name
        folder.mkdir(parents=True)
        for index in range(4):
            image = np.full((32, 32, 3), color + index, dtype=np.uint8)
            assert cv2.imwrite(str(folder / f"{index}.png"), image)


def _coco_multilabel_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    labels = ((1, 2), (1,), (2,), (1, 2), (1,), (2,))
    annotation_id = 1
    for index, image_labels in enumerate(labels, start=1):
        name = f"{index:012d}.jpg"
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :16] = (30 + index, 30 + index, 220 if 1 in image_labels else 30)
        image[:, 16:] = (220 if 2 in image_labels else 30, 30 + index, 30 + index)
        assert cv2.imwrite(str(root / name), image)
        images.append({"id": index, "file_name": name})
        for category_id in image_labels:
            annotations.append({
                "id": annotation_id, "image_id": index, "category_id": category_id,
                "bbox": [0, 0, 16, 32], "area": 512, "iscrowd": 0,
            })
            annotation_id += 1
    annotation_file = root / "instances_smoke.json"
    annotation_file.write_text(json.dumps({
        "images": images,
        "categories": [{"id": 1, "name": "red-object"}, {"id": 2, "name": "blue-object"}],
        "annotations": annotations,
    }), encoding="utf8")
    return annotation_file


def test_compact_classifier_dataset_is_split_per_class(tmp_path):
    _dataset(tmp_path)
    train, val, names = discover_training(tmp_path, val_fraction=0.25, seed=7)
    assert names == ["dark", "light"]
    assert len(train) == 6
    assert len(val) == 2
    assert {sample.label for sample in val} == {0, 1}


def test_classifier_forward_shape():
    model = TinyClassifier(classes=3, width=4)
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 3)


def test_classifier_uses_shared_multi_image_augmentations(tmp_path):
    _dataset(tmp_path)
    train, _, names = discover_training(tmp_path, val_fraction=0.25, seed=7)
    dataset = ImageDataset(train, 32, augment=True, num_classes=len(names), seed=3, augmentation={
        "mosaic": 1.0, "mosaic9": 0.0, "mixup": 1.0, "copy_paste": 1.0,
        "degrees": 10.0, "translate": 0.1, "scale": 0.2, "shear": 2.0,
        "perspective": 0.001, "flipud": 1.0, "fliplr": 1.0,
        "hsv_h": 0.01, "hsv_s": 0.2, "hsv_v": 0.2,
        "blur": 1.0, "noise": 1.0, "jpeg": 1.0, "erasing": 1.0,
        "square_crop": True,
    })
    image, target, _ = dataset[0]
    assert image.shape == (3, 32, 32)
    assert target.shape == (2,)
    assert torch.isclose(target.sum(), torch.tensor(1.0), atol=1e-5)
    assert (target >= 0).all()


def test_classifier_one_epoch_cpu_smoke(tmp_path):
    data = tmp_path / "data"
    output = tmp_path / "run"
    _dataset(data)
    assert main([
        "train", "task=classification", "arch=convnext-tiny", f"data={data}", f"output={output}", "epochs=1",
        "batch=2", "imgsz=32", "width=4", "device=cpu",
    ]) == 0
    assert (output / "best.pt").is_file()
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["format"] == "trainer-image-classifier-v4"
    assert checkpoint["arch"] == "convnext-tiny"
    assert checkpoint["classes"] == ["dark", "light"]
    assert (output / "weights" / "best.pt").is_file()
    assert (output / "logs" / "history.csv").is_file()
    assert (output / "reports" / "report.html").is_file()

    xai_output = tmp_path / "classifier_xai"
    assert main([
        "xai", "task=classification", f"source={data}",
        f"checkpoint={output / 'weights' / 'best.pt'}", f"output={xai_output}",
        "max_images=2", "activation_channels=2", "device=cpu",
    ]) == 0
    manifest = json.loads((xai_output / "xai" / "manifest.json").read_text())
    assert manifest["format"] == "trainer-classification-xai-v1"
    assert manifest["architecture"] == "convnext-tiny"
    assert len(manifest["images"]) == 2
    assert "red supports" in manifest["method"]
    assert (xai_output / manifest["images"][0]["overlay"]).is_file()
    assert (xai_output / manifest["images"][0]["activations"]).is_file()
    assert (xai_output / manifest["images"][0]["dossier"]).is_file()
    assert (xai_output / "reports" / "xai.html").is_file()


def test_coco_multilabel_classifier_cpu_smoke(tmp_path):
    data = _coco_multilabel_dataset(tmp_path / "coco")
    train_samples, val_samples, names = discover_training(data, val_fraction=0.33, seed=3)
    assert names == ["red-object", "blue-object"]
    assert classification_mode(train_samples + val_samples) == "multi-label"
    dataset = ImageDataset(train_samples, 32, augment=False, num_classes=len(names))
    _, target, _ = dataset[0]
    assert isinstance(target, torch.Tensor)
    assert target.shape == (2,)
    augmented = ImageDataset(train_samples, 32, augment=True, num_classes=len(names), seed=3, augmentation={
        "mosaic": 1.0, "mosaic9": 0.0, "mixup": 1.0, "copy_paste": 1.0,
        "degrees": 5.0, "translate": 0.05, "scale": 0.1, "fliplr": 1.0,
    })
    _, augmented_target, _ = augmented[0]
    assert augmented_target.shape == (2,)
    assert ((augmented_target >= 0) & (augmented_target <= 1)).all()

    output = tmp_path / "multilabel-run"
    assert main([
        "train", "task=classification", "arch=convnext-tiny", f"data={data}", f"output={output}",
        "epochs=1", "batch=2", "imgsz=32", "width=4", "workers=0", "device=cpu",
    ]) == 0
    checkpoint = torch.load(output / "weights" / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["classification_mode"] == "multi-label"
    assert checkpoint["activation"] == "sigmoid"
    assert checkpoint["metrics"]["classification_mode"] == "multi-label"
    assert "map" in checkpoint["metrics"]

    evaluation = tmp_path / "multilabel-eval"
    assert main([
        "evaluate", "task=classification", f"data={data}",
        f"checkpoint={output / 'weights' / 'best.pt'}", f"output={evaluation}",
        "batch=2", "device=cpu",
    ]) == 0
    metrics = json.loads((evaluation / "metrics.json").read_text())
    assert metrics["classification_mode"] == "multi-label"
    assert metrics["samples"] == 6
    assert metrics["confusion_matrix"] == []

    assert main([
        "tune-thresh", "task=classification", f"data={data}",
        f"checkpoint={output / 'weights' / 'best.pt'}", f"output={evaluation}",
        "batch=2", "steps=8", "device=cpu",
    ]) == 0
    thresholds = evaluation / "thresholds.json"
    assert set(json.loads(thresholds.read_text())["thresholds"]) == set(names)

    predictions = tmp_path / "multilabel-predict"
    assert main([
        "predict", "task=classification", f"source={data.parent}",
        f"checkpoint={output / 'weights' / 'best.pt'}", f"thresholds={thresholds}",
        f"output={predictions}", "batch=2", "topk=2", "device=cpu",
    ]) == 0
    rows = (predictions / "predictions.csv").read_text()
    assert "classes" in rows and "red-object" in rows

    xai_output = tmp_path / "multilabel-xai"
    assert main([
        "xai", "task=classification", f"source={data.parent / '000000000001.jpg'}",
        f"checkpoint={output / 'weights' / 'best.pt'}", f"output={xai_output}",
        "max_images=1", "activation_channels=2", "device=cpu",
    ]) == 0
    manifest = json.loads((xai_output / "xai" / "manifest.json").read_text())
    assert manifest["classification_mode"] == "multi-label"
    assert "predicted_labels" in manifest["images"][0]


def test_classifier_matches_detector_lifecycle_on_cpu(tmp_path):
    data = tmp_path / "data"
    run = tmp_path / "run"
    _dataset(data)
    common = ["batch=2", "imgsz=32", "width=4", "workers=0", "device=cpu"]
    assert main(["train", "task=classification", f"data={data}", f"output={run}", "epochs=1", *common]) == 0
    checkpoint = run / "weights" / "best.pt"

    evaluation = tmp_path / "evaluation"
    assert main(["evaluate", "task=classification", f"data={data}", f"checkpoint={checkpoint}",
                 f"output={evaluation}", "batch=2", "device=cpu"]) == 0
    assert (evaluation / "metrics.json").is_file()
    assert (evaluation / "per_image.csv").is_file()
    assert (evaluation / "reports" / "report.html").is_file()

    assert main(["tune-thresh", "task=classification", f"data={data}", f"checkpoint={checkpoint}",
                 f"output={evaluation}", "batch=2", "steps=8", "device=cpu"]) == 0
    thresholds = evaluation / "thresholds.json"
    assert thresholds.is_file()

    report = tmp_path / "report"
    assert main(["report", "task=classification", f"data={data}", f"checkpoint={checkpoint}",
                 f"thresholds={thresholds}", f"output={report}", "batch=2", "device=cpu"]) == 0
    assert (report / "reports" / "report.html").is_file()

    predictions = tmp_path / "predict"
    assert main(["predict", "task=classification", f"source={data}", f"checkpoint={checkpoint}",
                 f"thresholds={thresholds}", f"output={predictions}", "batch=2", "topk=2", "device=cpu"]) == 0
    assert (predictions / "predictions.csv").is_file()

    tools = tmp_path / "tools"
    assert main(["viz-augment", "task=classification", f"data={data}", f"output={tools}",
                 "imgsz=32", "n=4"]) == 0
    assert (tools / "plots" / "augment_preview.jpg").is_file()
    assert main(["bench-data", "task=classification", f"data={data}", f"output={tools}",
                 "imgsz=32", "batch=2", "batches=2"]) == 0
    assert (tools / "logs" / "benchmark.json").is_file()

    assert main(["train", "task=classification", f"data={data}", f"output={run}", "epochs=2",
                 f"resume={run / 'weights' / 'last.pt'}", *common]) == 0
    resumed = torch.load(run / "weights" / "last.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2

    finetune = tmp_path / "finetune"
    assert main(["finetune", "task=classification", f"data={data}", f"checkpoint={checkpoint}",
                 f"output={finetune}", "epochs=1", "freeze=1", *common]) == 0
    assert (finetune / "weights" / "best.pt").is_file()

    tuning = tmp_path / "tuning"
    assert main(["tune", "task=classification", f"data={data}", f"output={tuning}",
                 "iterations=1", "tune_epochs=1", *common]) == 0
    assert (tuning / "best_hyp.json").is_file()
