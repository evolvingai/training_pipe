import pytest
import torch

from trainer.detection.nn.model import DetectionModel, resolve_model_toml
from trainer.detection.nn.modules.conv import Conv
from trainer.detection.nn.registry import MODULES, get_module


def test_registry_has_builtins():
    for name in ("Conv", "RELAN", "A2Block", "CSP", "SPPF", "Detect", "Concat", "Upsample"):
        assert name in MODULES


def test_unknown_module_raises():
    with pytest.raises(KeyError):
        get_module("NoSuchModule")


def test_resolve_model_names():
    path, scale = resolve_model_toml("trainer-n")
    assert path.name == "trainer.toml" and scale == "n"
    path, scale = resolve_model_toml("trainer-csp-s")
    assert path.name == "trainer-csp.toml" and scale == "s"
    with pytest.raises(FileNotFoundError):
        resolve_model_toml("nonexistent-model")


@pytest.mark.parametrize("name", ["trainer-n", "trainer-csp-n", "trainer-fpn-n"])
def test_model_builds_and_forwards(name):
    model = DetectionModel(name, nc=5)
    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    x = torch.zeros(1, 3, 64, 64)
    model.train()
    feats = model(x)
    assert len(feats) == 3
    assert feats[0].shape == (1, 4 * 16 + 5, 8, 8)
    model.eval()
    with torch.no_grad():
        decoded, raw = model(x)
    assert decoded.shape[1] == 4 + 5
    # decoded scores are sigmoided
    assert decoded[:, 4:].min() >= 0 and decoded[:, 4:].max() <= 1


def test_scales_change_capacity():
    n = DetectionModel("trainer-n").num_params()
    s = DetectionModel("trainer-s").num_params()
    assert s > 2 * n


def test_fuse_preserves_output():
    model = DetectionModel("trainer-n", nc=2).eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        ref, _ = model(x)
        model.fuse()
        fused, _ = model(x)
    assert torch.allclose(ref, fused, atol=1e-4)
    assert not any(hasattr(m, "bn") for m in model.modules() if isinstance(m, Conv))


def test_transfer_load_skips_mismatched_head():
    src = DetectionModel("trainer-n", nc=80)
    dst = DetectionModel("trainer-n", nc=3)
    dst.load_weights(src.state_dict())
    # backbone weights transferred exactly
    src_w = dict(src.named_parameters())["layers.0.conv.weight"]
    dst_w = dict(dst.named_parameters())["layers.0.conv.weight"]
    assert torch.equal(src_w, dst_w)
