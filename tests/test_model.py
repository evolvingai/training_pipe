import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.detection.nn.model import DetectionModel, resolve_model_toml
from trainer.detection.nn.modules.conv import Conv
from trainer.detection.nn.modules.attention import A2C2f, AreaAttention, PaperAreaAttention
from trainer.detection.nn.registry import MODULES, get_module


def test_registry_has_builtins():
    for name in (
        "Conv", "RELAN", "A2Block", "C3k2", "A2C2f", "CSP", "SPPF",
        "Detect", "Concat", "Upsample",
    ):
        assert name in MODULES


def test_unknown_module_raises():
    with pytest.raises(KeyError):
        get_module("NoSuchModule")


def test_resolve_model_names():
    path, scale = resolve_model_toml("trainer-n")
    assert path.name == "trainer.toml" and scale == "n"
    path, scale = resolve_model_toml("trainer-csp-s")
    assert path.name == "trainer-csp.toml" and scale == "s"
    path, scale = resolve_model_toml("trainer-legacy-n")
    assert path.name == "trainer-legacy.toml" and scale == "n"
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
    sizes = [DetectionModel(f"trainer-{scale}").num_params() for scale in "nsmlx"]
    assert sizes == [2_602_288, 9_284_096, 20_199_168, 26_450_784, 59_210_784]


def test_default_model_uses_paper_topology():
    model = DetectionModel("trainer-n")
    assert len(model.layers) == 22
    assert isinstance(model.layers[2], MODULES["C3k2"])
    assert isinstance(model.layers[6], A2C2f)
    assert isinstance(model.layers[8], A2C2f)
    assert model.head.f == [14, 17, 20]


def test_long_scale_aliases():
    assert DetectionModel("trainer-nano").scale == "n"
    assert DetectionModel("trainer-small").scale == "s"
    assert DetectionModel("trainer-medium").scale == "m"
    assert DetectionModel("trainer-large").scale == "l"


def test_p2_variant_adds_stride_four_head():
    model = DetectionModel("trainer-p2-n", nc=3)
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    model.train()
    feats = model(torch.zeros(1, 3, 64, 64))
    assert [feature.shape[-1] for feature in feats] == [16, 8, 4, 2]


def test_stride_probe_does_not_mutate_batchnorm_stats():
    model = DetectionModel("trainer-n")
    for bn in (module for module in model.modules() if isinstance(module, nn.BatchNorm2d)):
        assert torch.equal(bn.running_mean, torch.zeros_like(bn.running_mean))
        assert torch.equal(bn.running_var, torch.ones_like(bn.running_var))
        assert bn.num_batches_tracked.item() == 0


@pytest.mark.parametrize("height", [8, 7])
def test_area_attention_preserves_area_layout(height):
    module = AreaAttention(8, num_heads=2, areas=4).eval()
    x = torch.randn(2, 8, height, 5)
    with torch.no_grad():
        actual = module(x)
        qkv = module.qkv(x)
        chunks = qkv.tensor_split(min(module.areas, height), dim=2)
        outputs = []
        for chunk in chunks:
            area_h = chunk.shape[2]
            q, k, v = chunk.reshape(2, 3, 2, 4, area_h, 5).unbind(1)
            q = q.permute(0, 1, 3, 4, 2).reshape(2, 2, area_h * 5, 4)
            k = k.permute(0, 1, 3, 4, 2).reshape(2, 2, area_h * 5, 4)
            v = v.permute(0, 1, 3, 4, 2).reshape(2, 2, area_h * 5, 4)
            out = F.scaled_dot_product_attention(q, k, v)
            outputs.append(out.reshape(2, 2, area_h, 5, 4).permute(0, 1, 4, 2, 3).reshape(2, 8, area_h, 5))
        expected = module.proj(torch.cat(outputs, 2) + module.pe(qkv[:, 16:]))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_area_attention_horizontal_shift_and_adaptive_partition():
    module = AreaAttention(
        16, num_heads=4, areas=2, axis="horizontal", shift=True, max_tokens=16,
    )
    x = torch.randn(2, 16, 7, 9, requires_grad=True)
    out = module(x)
    assert out.shape == x.shape
    out.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_paper_area_attention_preserves_shape_and_gradient():
    module = PaperAreaAttention(64, num_heads=2, area=4)
    x = torch.randn(2, 64, 8, 8, requires_grad=True)
    out = module(x)
    assert out.shape == x.shape
    out.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_paper_area_attention_rejects_invalid_partition():
    module = PaperAreaAttention(64, num_heads=2, area=4)
    with pytest.raises(ValueError, match="must be divisible"):
        module(torch.randn(1, 64, 3, 3))


def test_reg_max_override_changes_head_shape():
    model = DetectionModel("trainer-n", nc=3, reg_max=8).train()
    feats = model(torch.zeros(1, 3, 64, 64))
    assert feats[0].shape[1] == 4 * 8 + 3


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
