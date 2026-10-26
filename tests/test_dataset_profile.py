from __future__ import annotations

import json
from pathlib import Path

from automation.dataset_profile import build_profile


def test_coco_profile_builds_split_class_and_box_visualization_data(tmp_path: Path) -> None:
    annotations = tmp_path / "instances_val2017.json"
    annotations.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "cat.jpg"}, {"id": 2, "file_name": "empty.jpg"}],
        "categories": [{"id": 17, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 17, "bbox": [1, 2, 20, 30], "area": 600},
        ],
    }), encoding="utf8")

    profile = build_profile("detection", annotations)

    assert profile["format"] == "coco"
    assert profile["summary"] == {
        "samples": 2, "images": 2, "annotations": 1, "classes": 1,
        "labeled": 1, "unlabeled": 1, "invalid": 0, "crowd": 0,
    }
    assert profile["classes"][0]["name"] == "cat"
    assert profile["classes"][0]["splits"] == {"val2017": 1}
    assert profile["boxSizes"]["small"] == 1


def test_coco_multilabel_profile_counts_unique_image_class_pairs(tmp_path: Path) -> None:
    annotations = tmp_path / "instances_val2017.json"
    annotations.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "scene.jpg"}, {"id": 2, "file_name": "empty.jpg"}],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [5, 5, 4, 4]},
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [9, 9, 4, 4]},
        ],
    }), encoding="utf8")

    profile = build_profile("classification", annotations)

    assert profile["format"] == "coco-multilabel"
    assert profile["summary"]["samples"] == 2
    assert profile["summary"]["annotations"] == 2
    assert profile["summary"]["unlabeled"] == 1
    assert {item["name"]: item["total"] for item in profile["classes"]} == {"person": 1, "dog": 1}


def test_custom_class_folder_profile_uses_same_visualization_schema(tmp_path: Path) -> None:
    for split, class_name, count in (("train", "cat", 2), ("train", "dog", 1), ("val", "cat", 1)):
        folder = tmp_path / split / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (folder / f"{index}.jpg").write_bytes(b"test")

    profile = build_profile("classification", tmp_path)

    assert profile["schema"] == "trainer-dataset-profile-v1"
    assert profile["format"] == "class-folders"
    assert profile["summary"]["samples"] == 4
    assert [split["name"] for split in profile["splits"]] == ["train", "val"]
    assert {item["name"]: item["total"] for item in profile["classes"]} == {"cat": 3, "dog": 1}
