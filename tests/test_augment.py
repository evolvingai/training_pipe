import random

import numpy as np
import pytest

from trainer.augmentation import TRANSFORMS as SHARED_TRANSFORMS
from trainer.augmentation import Compose, load_augmentation_config
from trainer.detection.augment import TRANSFORMS, build_pipeline, pipeline_from_spec
from trainer.detection.augment.geometric import LetterBox, RandomFlip, RandomPerspective, SquareCrop
from trainer.detection.augment.photometric import GaussNoise, JPEGArtifacts, MotionBlur, RandomHSV
from trainer.detection.config import load_config


def make_sample(h=80, w=120):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[20:60, 30:90] = (0, 0, 255)
    boxes = np.array([[30.0, 20.0, 90.0, 60.0]], dtype=np.float32)
    cls = np.array([0.0], dtype=np.float32)
    return {"image": img, "boxes": boxes, "cls": cls}


def test_letterbox_preserves_aspect_and_maps_boxes():
    sample = LetterBox(64)(make_sample(), random.Random(0))
    assert sample["image"].shape == (64, 64, 3)
    r, (px, py) = sample["ratio"], sample["pad"]
    box = sample["boxes"][0]
    assert abs(box[0] - (30 * r + px)) < 1e-4
    assert abs(box[1] - (20 * r + py)) < 1e-4


def test_flip_involution():
    sample = make_sample()
    original = sample["boxes"].copy()
    flip = RandomFlip("horizontal", p=1.0)
    rng = random.Random(0)
    out = flip(flip(dict(sample, boxes=sample["boxes"].copy()), rng), rng)
    assert np.allclose(out["boxes"], original, atol=1e-4)


def test_flip_boxes_match_pixels():
    sample = make_sample()
    out = RandomFlip("horizontal", p=1.0)(sample, random.Random(0))
    x1, y1, x2, y2 = out["boxes"][0].astype(int)
    region = out["image"][y1:y2, x1:x2]
    assert (region[..., 2] > 200).mean() > 0.95  # red pixels inside the flipped box


def test_random_perspective_boxes_stay_valid():
    rng = random.Random(3)
    t = RandomPerspective(degrees=20, translate=0.1, scale=0.3, shear=5)
    for _ in range(20):
        out = t({**make_sample(), "boxes": make_sample()["boxes"].copy()}, rng)
        boxes = out["boxes"]
        h, w = out["image"].shape[:2]
        if boxes.shape[0]:
            assert (boxes[:, 2] > boxes[:, 0]).all()
            assert (boxes[:, 3] > boxes[:, 1]).all()
            assert boxes.min() >= 0 and boxes[:, [0, 2]].max() <= w and boxes[:, [1, 3]].max() <= h


def test_square_crop():
    out = SquareCrop()(make_sample(h=80, w=120), random.Random(0))
    assert out["image"].shape[0] == out["image"].shape[1] == 80


def test_photometric_keep_shape_and_boxes():
    for t in (RandomHSV(p=1.0), MotionBlur(p=1.0), GaussNoise(p=1.0), JPEGArtifacts(p=1.0)):
        sample = make_sample()
        before = sample["boxes"].copy()
        out = t(sample, random.Random(1))
        assert out["image"].shape == (80, 120, 3)
        assert np.array_equal(out["boxes"], before)


def test_registry_contains_all_builtins():
    for name in ("Mosaic", "MixUp", "CopyPaste", "RandomPerspective", "LetterBox",
                 "RandomFlip", "RandomHSV", "MotionBlur", "GaussNoise", "JPEGArtifacts",
                 "RandomErasing", "SquareCrop", "BrightnessContrast"):
        assert name in TRANSFORMS
    assert TRANSFORMS is SHARED_TRANSFORMS


def test_external_augmentation_profile(tmp_path):
    profile = tmp_path / "robust.toml"
    profile.write_text("[augmentation]\nmixup = 0.4\nblur = 0.2\nfliplr = 0.8\n")
    assert load_augmentation_config(profile) == {"mixup": 0.4, "blur": 0.2, "fliplr": 0.8}


def test_sequence_uses_identical_random_geometry():
    first = make_sample()
    second = make_sample()
    pipeline = Compose([RandomFlip("horizontal", p=0.5)], seed=1)
    output = pipeline.apply_many([first, second])
    assert np.array_equal(output[0]["image"], output[1]["image"])
    assert np.array_equal(output[0]["boxes"], output[1]["boxes"])


def test_pipeline_from_spec():
    pipeline = pipeline_from_spec([
        {"name": "LetterBox", "imgsz": 32},
        {"name": "RandomFlip", "direction": "horizontal", "p": 1.0},
    ], seed=0)
    out = pipeline(make_sample())
    assert out["image"].shape == (32, 32, 3)


def test_pipeline_from_spec_unknown_raises():
    with pytest.raises(KeyError):
        pipeline_from_spec([{"name": "NoSuchTransform"}])


def test_build_pipeline_eval_is_letterbox_only():
    cfg = load_config({"imgsz": 64})
    pipeline = build_pipeline(cfg, train=False)
    assert len(pipeline.transforms) == 1
    assert isinstance(pipeline.transforms[0], LetterBox)


def test_disable_transforms():
    cfg = load_config({"imgsz": 64, "mosaic": 1.0, "mixup": 0.5})
    pipeline = build_pipeline(cfg, train=True)
    pipeline.disable({"Mosaic", "MixUp"})
    by_name = {type(t).__name__: t for t in pipeline.transforms}
    assert by_name["Mosaic"].p == 0.0
    assert by_name["MixUp"].p == 0.0
