"""Export: fuse -> ONNX (torch.onnx), optional TensorRT engine, parity gates."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from trainer.detection.utils.logging import LOGGER

# parity tolerances (PRD): FP32 vs source, FP16 vs source
PARITY_ATOL = {"fp32": 1e-4, "fp16": 5e-3}


class _ExportWrapper(torch.nn.Module):
    """Eval-mode forward that returns only the decoded (B, 4+nc, A) tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        decoded, _ = self.model(x)
        return decoded


def export_onnx(model, imgsz: int = 640, opset: int = 17, dynamic: bool = False,
                half: bool = False, out_path: str | Path = "model.onnx",
                device: torch.device | None = None) -> Path:
    device = device or torch.device("cpu")
    model = copy.deepcopy(model).float().eval().to(device)
    model.fuse()
    wrapper = _ExportWrapper(model)
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=["images"], output_names=["output"],
        opset_version=opset, dynamic_axes=dynamic_axes, dynamo=False,
    )
    LOGGER.info(f"exported ONNX: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    if half:
        LOGGER.info("note: FP16 conversion is applied at TensorRT build time")
    return out_path


def verify_onnx_parity(model, onnx_path: str | Path, imgsz: int = 640, atol: float | None = None,
                       n_samples: int = 4, seed: int = 0) -> float:
    """Compare ONNX Runtime output against the source model. Returns max abs diff.

    Raises RuntimeError when the parity gate fails.
    """
    try:
        import onnxruntime as ort  # deferred: [deploy] extra
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for ONNX parity checks: pip install -e '.[deploy]'"
        ) from e

    atol = atol if atol is not None else PARITY_ATOL["fp32"]
    model = copy.deepcopy(model).float().eval().cpu()
    model.fuse()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    for _ in range(n_samples):
        x = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)
        with torch.no_grad():
            ref, _ = model(torch.from_numpy(x))
        out = sess.run(None, {"images": x})[0]
        max_diff = max(max_diff, float(np.abs(ref.numpy() - out).max()))
    if max_diff > atol:
        raise RuntimeError(f"ONNX parity gate failed: max abs diff {max_diff:.2e} > atol {atol:.0e}")
    LOGGER.info(f"ONNX parity OK: max abs diff {max_diff:.2e} (atol {atol:.0e})")
    return max_diff


def export_tensorrt(onnx_path: str | Path, half: bool = False, int8: bool = False,
                    workspace_gb: int = 4, calib_images: list[str] | None = None,
                    imgsz: int = 640, out_path: str | Path | None = None) -> Path:
    """Build a TensorRT engine from an ONNX file ([deploy] extra, needs a GPU).

    Compatible with TensorRT 8.x and 10.x:
    - TRT 10 removed the EXPLICIT_BATCH network flag (now always on by default).
    - TRT 10 removed builder.platform_has_fast_fp16/int8; precision flags are set
      unconditionally and TRT handles capability checking internally.
    """
    try:
        import tensorrt as trt  # deferred: [deploy] extra
    except ImportError as e:
        raise ImportError(
            "tensorrt is required for engine export: pip install -e '.[deploy]'"
        ) from e

    trt_version = tuple(int(x) for x in trt.__version__.split(".")[:2])
    LOGGER.info(f"TensorRT version: {trt.__version__}")

    onnx_path = Path(onnx_path)
    out_path = Path(out_path) if out_path else onnx_path.with_suffix(".engine")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # EXPLICIT_BATCH flag was required in TRT 8; removed in TRT 10 (always on).
    _explicit_batch = getattr(trt, "NetworkDefinitionCreationFlag", None)
    _explicit_batch = getattr(_explicit_batch, "EXPLICIT_BATCH", None)
    if _explicit_batch is not None:
        network = builder.create_network(1 << int(_explicit_batch))
    else:
        network = builder.create_network()

    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    # platform_has_fast_fp16/int8 removed in TRT 10; fall back to True (TRT handles internally).
    has_fp16 = getattr(builder, "platform_has_fast_fp16", True)
    has_int8 = getattr(builder, "platform_has_fast_int8", True)

    if half and has_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8 and has_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calib_images:
            config.int8_calibrator = _EntropyCalibrator(calib_images, imgsz,
                                                        cache=str(onnx_path.with_suffix(".calib")))
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    out_path.write_bytes(engine)
    LOGGER.info(f"exported TensorRT engine: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def _make_entropy_calibrator_class():
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "tensorrt is required for INT8 calibration: pip install -e '.[deploy]'"
        ) from e

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, image_paths: list[str], imgsz: int, cache: str, batch: int = 8):
            super().__init__()
            import cv2
            try:
                import pycuda  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "pycuda is required for INT8 calibration: pip install -e '.[deploy]'"
                ) from e

            self.cv2 = cv2
            self.paths = image_paths
            self.imgsz = imgsz
            self.batch = batch
            self.cache_path = Path(cache)
            self.index = 0
            self.device_mem = None

        def get_batch_size(self):
            return self.batch

        def get_batch(self, names):
            try:
                import pycuda.autoinit  # noqa: F401
                import pycuda.driver as cuda
            except ImportError as e:
                raise ImportError(
                    "pycuda is required for INT8 calibration: pip install -e '.[deploy]'"
                ) from e

            if self.index + self.batch > len(self.paths):
                return None
            imgs = []
            for p in self.paths[self.index:self.index + self.batch]:
                img = self.cv2.imread(p)
                img = self.cv2.resize(img, (self.imgsz, self.imgsz))
                imgs.append(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            self.index += self.batch
            arr = np.ascontiguousarray(np.stack(imgs))
            if self.device_mem is None:
                self.device_mem = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(self.device_mem, arr)
            return [int(self.device_mem)]

        def read_calibration_cache(self):
            return self.cache_path.read_bytes() if self.cache_path.exists() else None

        def write_calibration_cache(self, cache):
            self.cache_path.write_bytes(cache)

    return EntropyCalibrator


def _EntropyCalibrator(image_paths: list[str], imgsz: int, cache: str):
    return _make_entropy_calibrator_class()(image_paths, imgsz, cache)
