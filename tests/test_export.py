import importlib.util

import pytest
import torch

from trainer.detection.api import Model
from trainer.detection.engine.exporter import _ExportWrapper, export_onnx, verify_onnx_parity
from trainer.detection.nn.model import DetectionModel

HAS_ONNX = importlib.util.find_spec("onnx") is not None
HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


def test_export_wrapper_returns_single_tensor():
    model = DetectionModel("trainer-n", nc=2).eval()
    out = _ExportWrapper(model)(torch.zeros(1, 3, 64, 64))
    assert isinstance(out, torch.Tensor)
    assert out.shape[1] == 6  # 4 + nc


def test_model_export_accepts_explicit_output(monkeypatch, tmp_path):
    expected = tmp_path / "deployment" / "model.onnx"

    def fake_export(_model, **kwargs):
        assert kwargs["out_path"] == str(expected)
        return expected

    monkeypatch.setattr("trainer.detection.engine.exporter.export_onnx", fake_export)
    monkeypatch.setattr("trainer.detection.engine.exporter.verify_onnx_parity", lambda *_args, **_kwargs: 0.0)
    assert Model("trainer-n").export(format="onnx", out=expected.with_suffix(""), imgsz=64) == expected


@pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")
def test_onnx_export(tmp_path):
    model = DetectionModel("trainer-n", nc=2)
    path = export_onnx(model, imgsz=64, out_path=tmp_path / "m.onnx")
    assert path.exists() and path.stat().st_size > 1000


@pytest.mark.skipif(not (HAS_ONNX and HAS_ORT), reason="onnx/onnxruntime not installed")
def test_onnx_parity_gate(tmp_path):
    model = DetectionModel("trainer-n", nc=2)
    path = export_onnx(model, imgsz=64, out_path=tmp_path / "m.onnx")
    diff = verify_onnx_parity(model, path, imgsz=64)
    assert diff < 1e-4
