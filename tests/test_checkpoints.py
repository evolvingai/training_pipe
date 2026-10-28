from pathlib import Path

import torch

from trainer.detection.config import load_toml
from trainer.detection.nn.model import DetectionModel, resolve_model_toml
from trainer.detection.utils.checkpoints import (
    CHECKPOINT_VERSION, atomic_torch_save, checkpoint_safe, load_checkpoint, model_from_checkpoint,
)


def test_portable_custom_architecture_checkpoint(tmp_path):
    cfg = load_toml(resolve_model_toml("trainer-n")[0])
    cfg["backbone"][0][3][0] = 48
    model = DetectionModel(cfg, nc=2, reg_max=8)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "model_cfg": model.architecture_spec(),
        "model_name": model.model_name,
        "nc": model.nc,
        "ch": model.ch,
        "reg_max": model.head.reg_max,
        "cfg": checkpoint_safe({"project": Path("runs")}),
    }
    path = tmp_path / "custom.pt"
    torch.save(checkpoint, path)

    loaded = load_checkpoint(path)
    rebuilt = model_from_checkpoint(loaded)
    rebuilt.load_state_dict(loaded["model"])
    assert rebuilt.architecture_spec() == model.architecture_spec()
    assert rebuilt.head.reg_max == 8
    assert loaded["cfg"]["project"] == "runs"


def test_rejects_newer_checkpoint_version(tmp_path):
    path = tmp_path / "future.pt"
    torch.save({"checkpoint_version": CHECKPOINT_VERSION + 1}, path)
    try:
        load_checkpoint(path)
    except RuntimeError as error:
        assert "newer than supported" in str(error)
    else:
        raise AssertionError("future checkpoint should be rejected")


def test_atomic_checkpoint_save(tmp_path):
    path = tmp_path / "weights" / "last.pt"
    atomic_torch_save({"checkpoint_version": CHECKPOINT_VERSION, "value": torch.tensor([3])}, path)
    assert load_checkpoint(path)["value"].item() == 3
    assert list(path.parent.glob(".last.pt.*.tmp")) == []
