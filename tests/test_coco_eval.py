import importlib.util
import json

import pytest

from trainer.detection.data.coco_sota_fixes import coco_evaluate


@pytest.mark.skipif(importlib.util.find_spec("pycocotools") is None, reason="pycocotools not installed")
def test_official_coco_evaluator_handles_empty_predictions(tmp_path):
    annotations = tmp_path / "instances.json"
    annotations.write_text(json.dumps({
        "info": {},
        "licenses": [],
        "images": [{"id": 42, "file_name": "one.jpg", "height": 32, "width": 32}],
        "categories": [{"id": 7, "name": "thing", "supercategory": "object"}],
        "annotations": [{
            "id": 1, "image_id": 42, "category_id": 7, "bbox": [8, 8, 12, 12],
            "area": 144, "iscrowd": 0,
        }],
    }))
    result = coco_evaluate(annotations, [], image_ids=[42])
    assert result["AP"] == 0.0
    assert result["AP50"] == 0.0


def _annotations(tmp_path, count=1, crowd=False):
    annotations = tmp_path / "instances.json"
    boxes = [[2 * i, 0, 1, 1] for i in range(count)]
    data = {
        "info": {}, "images": [{"id": 42, "file_name": "one.jpg", "height": 32, "width": 256}],
        "categories": [{"id": 7, "name": "thing"}],
        "annotations": [{"id": i + 1, "image_id": 42, "category_id": 7, "bbox": b,
                         "area": 1, "iscrowd": 0} for i, b in enumerate(boxes)],
    }
    if crowd:
        data["annotations"].append({"id": count + 1, "image_id": 42, "category_id": 7,
                                    "bbox": [0, 20, 20, 10], "area": 200, "iscrowd": 1})
    annotations.write_text(json.dumps(data))
    predictions = [{"image_id": 42, "category_id": 7, "bbox": b, "score": .9 - i * .001}
                   for i, b in enumerate(boxes)]
    return annotations, predictions


@pytest.mark.skipif(importlib.util.find_spec("pycocotools") is None, reason="pycocotools not installed")
def test_perfect_standard_and_custom_cap(tmp_path):
    annotations, predictions = _annotations(tmp_path, count=110)
    standard = coco_evaluate(annotations, predictions)
    custom = coco_evaluate(annotations, predictions, max_dets=[1, 10, 300])
    assert 0 < standard["AP"] < 1 and standard["maxDets"] == 100
    assert custom["AP"] == pytest.approx(1.) and custom["maxDets"] == 300


@pytest.mark.skipif(importlib.util.find_spec("pycocotools") is None, reason="pycocotools not installed")
def test_crowd_region_detection_ignored_by_official_evaluator(tmp_path):
    annotations, predictions = _annotations(tmp_path, crowd=True)
    predictions.insert(0, {"image_id": 42, "category_id": 7, "bbox": [1, 21, 2, 2], "score": .99})
    assert coco_evaluate(annotations, predictions)["AP"] == pytest.approx(1.)


@pytest.mark.skipif(importlib.util.find_spec("pycocotools") is None, reason="pycocotools not installed")
def test_custom_matches_official_on_ordinary_fixture(tmp_path):
    import torch
    from trainer.detection.utils.metrics import DetMetrics
    annotations, predictions = _annotations(tmp_path)
    metrics = DetMetrics(1)
    metrics.update(torch.tensor([[0., 0., 1., 1., .9, 0.]]),
                   torch.tensor([[0., 0., 1., 1.]]), torch.zeros(1))
    assert metrics.compute()["mAP50-95"] == pytest.approx(coco_evaluate(annotations, predictions)["AP"])
