import pytest

from trainer.detection.config import dump, dumps, load_config, load_toml, merge_overrides


def test_default_config_loads():
    cfg = load_config()
    assert cfg["imgsz"] == 640
    assert cfg["task"] == "detect"


def test_override_merge_and_coercion():
    cfg = load_config({"epochs": "50", "lr0": "0.02", "amp": "false"})
    assert cfg["epochs"] == 50 and isinstance(cfg["epochs"], int)
    assert cfg["lr0"] == 0.02 and isinstance(cfg["lr0"], float)
    assert cfg["amp"] is False


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        load_config({"epochz": 10})


def test_musgd_optimizer_and_blend_weights():
    cfg = load_config({"optimizer": "musgd", "muon_weight": "0.5", "sgd_weight": "0.75"})
    assert cfg["optimizer"] == "musgd"
    assert cfg["muon_weight"] == 0.5 and isinstance(cfg["muon_weight"], float)
    assert cfg["sgd_weight"] == 0.75


@pytest.mark.parametrize("overrides", [
    {"task": "classify"},
    {"batch": 0},
    {"fraction": 1.1},
    {"mosaic": -0.1},
    {"cache": "maybe"},
    {"optimizer": "muon"},
    {"muon_weight": -0.1},
    {"sgd_weight": -1.0},
])
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        load_config(overrides)


def test_non_strict_merge_accepts_new_keys():
    out = merge_overrides({"a": 1}, {"b": 2}, strict=False)
    assert out == {"a": 1, "b": 2}


def test_toml_writer_roundtrip(tmp_path):
    data = {
        "name": "exp",
        "lr0": 0.01,
        "epochs": 100,
        "amp": True,
        "scales": {"n": [0.33, 0.25, 1024]},
        "tags": ["a", "b"],
        "note": 'quote " and backslash \\',
    }
    path = tmp_path / "out.toml"
    dump(data, path)
    loaded = load_toml(path)
    assert loaded == data


def test_dumps_bool_and_float_format():
    text = dumps({"flag": False, "value": 1.5})
    assert "flag = false" in text
    assert "value = 1.5" in text
