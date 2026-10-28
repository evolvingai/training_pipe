"""Reproducible model-profile benchmark for capacity and inference latency."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from trainer.detection.nn.model import DetectionModel
from trainer.detection.utils.checks import cuda_device_index
from trainer.detection.utils.logging import LOGGER


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(cuda_device_index(device))


def benchmark_models(model_base: str, scales: list[str], imgsz: int, batch: int,
                     device: torch.device, warmup: int = 10, iterations: int = 50) -> dict:
    """Benchmark one architecture across named capacity scales."""
    if iterations <= 0 or warmup < 0:
        raise ValueError("benchmark iterations must be positive and warmup non-negative")
    rows = []
    for scale in scales:
        spec = f"{model_base}-{scale}"
        model = DetectionModel(spec).to(device).eval()
        sample = torch.zeros(batch, model.ch, imgsz, imgsz, device=device)
        with torch.inference_mode():
            for _ in range(warmup):
                model(sample)
            _synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
            samples = []
            for _ in range(iterations):
                started = time.perf_counter()
                model(sample)
                _synchronize(device)
                samples.append((time.perf_counter() - started) * 1000.0)
        p95_index = max(int(round(0.95 * len(samples))) - 1, 0)
        p95 = sorted(samples)[p95_index]
        median = statistics.median(samples)
        peak = (torch.cuda.max_memory_allocated(cuda_device_index(device)) / 2**20
                if device.type == "cuda" else None)
        row = {
            "model": spec,
            "parameters": model.num_params(),
            "strides": [int(v) for v in model.stride.tolist()],
            "batch": batch,
            "imgsz": imgsz,
            "latency_ms_p50": median,
            "latency_ms_p95": p95,
            "throughput_images_s": batch * 1000.0 / median,
            "peak_cuda_mib": peak,
        }
        rows.append(row)
        LOGGER.info(
            f"{spec}: {row['parameters']:,} params, p50={median:.2f} ms, "
            f"p95={p95:.2f} ms, {row['throughput_images_s']:.1f} img/s"
        )
        del model, sample
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "device": str(device), "torch": torch.__version__, "warmup": warmup,
        "iterations": iterations, "profiles": rows,
    }


def write_benchmark(result: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def benchmark_checkpoint(model, paths: list[str], cfg: dict, device: torch.device,
                         warmup: int = 10, iterations: int = 50) -> dict:
    """Actual checkpoint: RGB read -> letterbox -> transfer -> forward -> NMS -> CPU boxes.

    Disk reads are included (OS cache may be warm); plotting and file writes are
    excluded. Each component is synchronized, so reported overhead is not hidden.
    """
    from trainer.detection.engine.experiments import StepProfiler
    from trainer.detection.engine.predictor import Predictor
    from trainer.detection.utils.boxes import scale_boxes
    from trainer.detection.utils.image import imread_rgb
    from trainer.detection.utils.nms import non_max_suppression

    if not paths or iterations < 1 or warmup < 0 or int(cfg["batch"]) < 1:
        raise ValueError("benchmark needs images, positive batch/iterations and nonnegative warmup")
    predictor = Predictor(model.to(device).eval(), cfg, device=device)
    profiler = StepProfiler(True, device)
    samples = []
    batch_size = int(cfg["batch"])
    with torch.inference_mode():
        for iteration in range(warmup + iterations):
            chosen = [paths[(iteration * batch_size + i) % len(paths)] for i in range(batch_size)]
            if iteration == warmup and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
            profiler.reset()
            with profiler.measure("end_to_end"):
                with profiler.measure("read_preprocess"):
                    tensors, metas = [], []
                    for path in chosen:
                        original = imread_rgb(path)
                        if original is None:
                            raise ValueError(f"unreadable benchmark image: {path}")
                        tensor, ratio, pad = predictor._preprocess(original)
                        tensors.append(tensor)
                        metas.append((original.shape[:2], ratio, pad))
                with profiler.measure("transfer"):
                    batch = torch.stack(tensors).to(device).float() / 255.
                with profiler.measure("forward"):
                    decoded, _ = model(batch)
                with profiler.measure("nms"):
                    predictions = non_max_suppression(decoded.float(), conf_thres=float(cfg["val_conf"]),
                                                     iou_thres=float(cfg["iou"]), max_det=int(cfg["max_det"]),
                                                     multi_label=bool(cfg["nms_multi_label"]))
                with profiler.measure("cpu_rescale"):
                    for prediction, (shape, ratio, pad) in zip(predictions, metas):
                        cpu = prediction.cpu()
                        scale_boxes(cpu[:, :4], batch.shape[-2:], shape, pad=pad, gain=ratio)
            if iteration >= warmup:
                samples.append(dict(profiler.values))
    timing = {}
    for key in samples[0]:
        values = sorted(sample[key] * 1000 for sample in samples)
        timing[key] = {"p50_ms": statistics.median(values),
                       "p95_ms": values[max(int(.95 * len(values) + .999999) - 1, 0)]}
    return {"parameters": model.num_params(), "batch": batch_size, "imgsz": cfg["imgsz"],
            "device": str(device), "precision": "float32", "warmup": warmup, "iterations": iterations,
            "images_available": len(paths), "score_floor": cfg["val_conf"], "iou": cfg["iou"],
            "multi_label": cfg["nms_multi_label"], "max_det": cfg["max_det"], "timings": timing,
            "throughput_images_s": batch_size * 1000 / timing["end_to_end"]["p50_ms"],
            "peak_cuda_bytes": (torch.cuda.max_memory_allocated(cuda_device_index(device))
                                if device.type == "cuda" else None),
            "scope": "synchronized end-to-end including image reads; OS cache may be warm; no plotting/writes"}
