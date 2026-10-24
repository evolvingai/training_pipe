"""Built-in task plugins and adapters to the shared Trainer CLI flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from trainer.registry import register_task, tasks

TASK_ALIASES = {
    "detect": "detection", "detector": "detection",
    "classification": "classification", "image": "classification", "image-classification": "classification",
    "video": "video-classification", "action": "video-classification", "action-recognition": "video-classification",
    "video-classification": "video-classification", "vlm": "vlm",
}

AUGMENTATION_OPTIONS = (
    "augmentation_config", "aug_profile", "mosaic", "mosaic9", "mixup", "mixup_alpha",
    "copy_paste", "copy_paste_iou", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v", "brightness_contrast", "brightness",
    "contrast", "blur", "noise", "jpeg", "erasing", "square_crop", "square_crop_jitter",
    "min_box_visibility",
)


def resolve_task_name(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    return TASK_ALIASES.get(key, key)


def _output(params: dict[str, Any], fallback: str) -> Path:
    if params.get("output"):
        return Path(str(params["output"]))
    return Path(str(params.get("project", "runs"))) / str(params.get("name", fallback))


def _option(argv: list[str], name: str, value: Any, *, boolean_optional: bool = False) -> None:
    flag = f"--{name.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        elif boolean_optional:
            argv.append(f"--no-{name.replace('_', '-')}")
        return
    if value is not None and value != "":
        argv.extend([flag, str(value)])


def _options(argv: list[str], params: dict[str, Any], names: Iterable[str]) -> None:
    for name in names:
        if name in params:
            _option(argv, name, params[name])


@register_task("detection", actions=("train", "finetune", "val", "predict", "export", "tune", "tune-thresh",
                                                "report", "viz-augment", "bench-data"),
               default_architecture="trainer-n", description="Anchor-free Trainer detector.")
def run_detection(action: str, params: dict[str, Any]) -> int:
    from trainer.detection.cli import main as detector_main
    values = dict(params)
    architecture = values.pop("arch", None)
    if architecture and "model" not in values:
        values["model"] = architecture
    encoded = (f"{key}={str(value).lower() if isinstance(value, bool) else value}"
               for key, value in values.items())
    argv = [action, *encoded]
    return detector_main(argv)


@register_task("classification", actions=("train", "finetune", "evaluate", "val", "predict", "export", "tune",
                                                  "tune-thresh", "report", "viz-augment", "bench-data", "xai"),
               default_architecture="convnext-tiny",
               description="Full-lifecycle folder-based image classification.")
def run_classification(action: str, params: dict[str, Any]) -> int:
    from trainer.classification.engine import main as classifier_main
    action = "evaluate" if action == "val" else action
    output = _output(params, f"classification_{action}")
    argv = [action]
    if action in {"train", "finetune", "evaluate", "tune", "tune-thresh", "report", "viz-augment", "bench-data"}:
        _option(argv, "data", params.get("data"))
    if action in {"predict", "xai"}:
        _option(argv, "source", params.get("source"))
    if action in {"finetune", "evaluate", "predict", "xai", "export", "tune-thresh", "report"}:
        _option(argv, "checkpoint", params.get("checkpoint") or params.get("model"))
    export_format = str(params.get("format", "onnx")).lower()
    suffix = {
        "evaluate": "metrics.json", "predict": "predictions.csv", "export": "model.onnx",
        "tune-thresh": "thresholds.json", "report": "reports/report.html",
        "viz-augment": "plots/augment_preview.jpg", "bench-data": "logs/benchmark.json",
    }.get(action)
    if action == "export" and export_format in {"tensorrt", "trt", "engine"}:
        suffix = "model.engine"
    _option(argv, "output", output if suffix is None else output / suffix)
    if action in {"train", "finetune", "tune"}:
        _options(argv, params, (
            "arch", "epochs", "batch", "imgsz", "width", "dropout", "optimizer", "lr",
            "final_lr_fraction", "momentum", "weight_decay", "label_smoothing", "val_fraction", "seed",
            "workers", "device", "accumulate", "grad_clip", "ema_decay", "ema_warmup_steps", "patience",
            "min_delta", "min_epochs", "freeze", "save_period", "resume", "mlflow_tracking_uri",
            "mlflow_experiment", "mlflow_run_name",
            *AUGMENTATION_OPTIONS,
        ))
        for option in ("amp", "compile", "channels_last", "mlflow"):
            if option in params:
                _option(argv, option, params[option], boolean_optional=True)
        if action == "tune":
            _options(argv, params, ("iterations", "tune_epochs"))
    elif action in {"evaluate", "tune-thresh", "report"}:
        _options(argv, params, ("batch", "workers", "device"))
        if action == "tune-thresh":
            _option(argv, "steps", params.get("steps"))
        if action == "report":
            _option(argv, "thresholds", params.get("thresholds"))
    elif action == "predict":
        _options(argv, params, ("batch", "workers", "device", "conf", "thresholds", "topk"))
    elif action == "xai":
        _options(argv, params, ("device", "target", "max_images", "intensity", "activation_channels"))
    elif action == "export":
        _options(argv, params, ("device", "format", "opset", "workspace"))
        if "half" in params:
            _option(argv, "half", params["half"], boolean_optional=True)
    elif action == "viz-augment":
        _options(argv, params, ("imgsz", "n", "val_fraction", "seed", *AUGMENTATION_OPTIONS))
    elif action == "bench-data":
        _options(argv, params, ("imgsz", "batch", "workers", "batches", "val_fraction", "seed",
                                *AUGMENTATION_OPTIONS))
    return classifier_main(argv)


@register_task("video-classification", actions=("train", "evaluate", "val", "predict", "export"),
               default_architecture="r2plus1d-tiny", description="Clip-based video/action classification.")
def run_video(action: str, params: dict[str, Any]) -> int:
    from automation.video_classifier import main as video_main
    action = "evaluate" if action == "val" else action
    output = _output(params, f"video_{action}")
    argv = [action]
    if action in {"train", "evaluate"}:
        _option(argv, "data", params.get("data"))
    if action == "predict":
        _option(argv, "source", params.get("source"))
    if action != "train":
        _option(argv, "checkpoint", params.get("checkpoint") or params.get("model"))
    export_format = str(params.get("format", "onnx")).lower()
    export_suffix = "model.engine" if export_format in {"tensorrt", "trt", "engine"} else "model.onnx"
    suffix = export_suffix if action == "export" else "metrics.json" if action == "evaluate" else "predictions.csv"
    _option(argv, "output", output if action == "train" else output / suffix)
    _options(argv, params, ("arch", "epochs", "batch", "imgsz", "clip_len", "width", "hidden", "dropout", "lr",
                            "weight_decay", "label_smoothing", "val_fraction", "seed", "workers", "device", "opset",
                            "format", "workspace", *AUGMENTATION_OPTIONS))
    if action == "export" and "half" in params:
        _option(argv, "half", params["half"], boolean_optional=True)
    return video_main(argv)


@register_task("vlm", actions=("tune", "train", "merge", "export"), default_architecture="hf-lora",
               description="Multimodal image-text PEFT/LoRA tuning.")
def run_vlm(action: str, params: dict[str, Any]) -> int:
    if action == "export":
        from automation.vlm_export import main as vlm_export_main

        export_format = str(params.get("format", "onnx")).lower()
        suffix = "model.engine" if export_format in {"tensorrt", "trt", "engine"} else "model.onnx"
        output = _output(params, "vlm_export") / suffix
        argv: list[str] = []
        _option(argv, "model", params.get("model"))
        _option(argv, "adapter", params.get("adapter") or params.get("checkpoint"))
        _option(argv, "output", output)
        _option(argv, "format", "tensorrt" if export_format in {"tensorrt", "trt", "engine"} else "onnx")
        _options(argv, params, ("device", "dtype", "sample_image", "prompt", "image_size", "max_length",
                                "opset", "workspace"))
        for option in ("half", "trust_remote_code"):
            if option in params:
                _option(argv, option, params[option], boolean_optional=True)
        return vlm_export_main(argv)

    from automation.vlm_tune import main as vlm_main
    action = "tune" if action == "train" else action
    arch = str(params.get("arch", "hf-lora")).lower().replace("_", "-")
    if arch != "hf-lora":
        raise ValueError(
            "dataset VLM tuning uses arch=hf-lora; use trainer smoke task=vlm "
            "arch=tiny-vlm-lora for offline validation"
        )
    output = _output(params, f"vlm_{action}")
    argv = [action]
    _option(argv, "model", params.get("model"))
    _option(argv, "output", output)
    if action == "tune":
        _option(argv, "data", params.get("data"))
        _options(argv, params, ("epochs", "batch", "lr", "weight_decay", "max_length", "max_samples", "lora_r",
                                "lora_alpha", "lora_dropout", "target_modules", "gradient_accumulation",
                                "max_grad_norm", "log_every", "seed", "imgsz", *AUGMENTATION_OPTIONS))
        if "gradient_checkpointing" in params:
            _option(argv, "gradient_checkpointing", params["gradient_checkpointing"], boolean_optional=True)
    else:
        _option(argv, "adapter", params.get("adapter") or params.get("checkpoint"))
    _options(argv, params, ("device", "dtype"))
    if params.get("trust_remote_code"):
        _option(argv, "trust_remote_code", True)
    return vlm_main(argv)


def run_task(task: str, action: str, params: dict[str, Any]) -> int:
    plugin = tasks.get(resolve_task_name(task))
    canonical_action = action.strip().lower().replace("_", "-")
    if canonical_action not in plugin.actions:
        raise ValueError(f"task '{plugin.name}' does not support '{action}'; actions: {', '.join(plugin.actions)}")
    values = dict(params)
    return plugin.runner(canonical_action, values)
