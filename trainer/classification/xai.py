"""Signed Grad-CAM and feature-activation evidence for image classifiers."""

from __future__ import annotations

import csv
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn

from trainer.classification.data import ImageDataset, Sample


def _feature_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return spatial feature layers from shallow to deep."""
    if hasattr(model, "stages"):
        stages = list(model.stages)
        if stages:
            return [(f"stage_{index + 1}", stage) for index, stage in enumerate(stages)]
    if hasattr(model, "features"):
        return [("features", model.features)]
    convolutions = [(name, module) for name, module in model.named_modules()
                    if isinstance(module, nn.Conv2d)]
    if convolutions:
        return [convolutions[-1]]
    raise ValueError("classifier has no spatial feature layer suitable for Grad-CAM")


def _target_index(value: str, names: list[str], probabilities: torch.Tensor) -> int:
    target = str(value or "predicted").strip()
    if target.lower() in {"predicted", "top", "auto"}:
        return int(probabilities.argmax())
    if target in names:
        return names.index(target)
    try:
        index = int(target)
    except ValueError as error:
        raise ValueError(f"target must be predicted, a class name, or an index; classes: {names}") from error
    if index < 0 or index >= len(names):
        raise ValueError(f"target class index {index} is outside 0..{len(names) - 1}")
    return index


def _normalise_signed(value: np.ndarray) -> np.ndarray:
    scale = max(float(np.abs(value).max()), 1e-12)
    return np.clip(value / scale, -1.0, 1.0).astype(np.float32)


def _signed_overlay(image: np.ndarray, signed: np.ndarray, intensity: float) -> np.ndarray:
    height, width = image.shape[:2]
    cam = cv2.resize(signed, (width, height), interpolation=cv2.INTER_CUBIC)
    positive = np.clip(cam, 0.0, 1.0)
    negative = np.clip(-cam, 0.0, 1.0)
    heat = np.zeros_like(image, dtype=np.float32)
    heat[..., 2] = positive * 255.0  # red: supports selected class
    heat[..., 0] = negative * 255.0  # blue: opposes selected class
    alpha = np.clip(np.abs(cam) * intensity, 0.0, 1.0)[..., None]
    return np.clip(image.astype(np.float32) * (1.0 - alpha) + heat * alpha, 0, 255).astype(np.uint8)


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(result)


def _label(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    header = 52 if subtitle else 34
    canvas = np.full((image.shape[0] + header, image.shape[1], 3), 18, np.uint8)
    canvas[header:] = image
    cv2.putText(canvas, title[:58], (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (245, 245, 245), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle[:82], (10, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                    (175, 185, 195), 1, cv2.LINE_AA)
    return canvas


def _activation_montage(captures: list[tuple[str, torch.Tensor]], channels: int = 4) -> np.ndarray:
    tile = 104
    rows: list[np.ndarray] = []
    for layer_name, activation in captures:
        values = activation[0].detach().float().cpu()
        energy = values.abs().mean(dim=(1, 2))
        selected = energy.topk(min(channels, values.shape[0])).indices.tolist()
        tiles: list[np.ndarray] = []
        for channel in selected:
            feature = values[channel].numpy()
            low, high = float(feature.min()), float(feature.max())
            normalised = np.zeros_like(feature, dtype=np.uint8) if high <= low else (
                (feature - low) * (255.0 / (high - low))
            ).astype(np.uint8)
            coloured = cv2.applyColorMap(cv2.resize(normalised, (tile, tile), interpolation=cv2.INTER_CUBIC),
                                         cv2.COLORMAP_VIRIDIS)
            cv2.rectangle(coloured, (0, 0), (tile, 20), (8, 12, 18), -1)
            cv2.putText(coloured, f"{layer_name} c{channel}", (5, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.34, (245, 245, 245), 1, cv2.LINE_AA)
            tiles.append(coloured)
        while len(tiles) < channels:
            tiles.append(np.full((tile, tile, 3), 18, np.uint8))
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0)


def _token(index: int, path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "image"
    return f"{index:04d}_{stem[:70]}"


def _write_gallery(output: Path, rows: list[dict[str, Any]], architecture: str) -> Path:
    cards = "".join(
        f"""<article><a href="../xai/dossiers/{html.escape(row['token'])}_dossier.jpg"><img src="../xai/dossiers/{html.escape(row['token'])}_dossier.jpg" alt="xAI dossier"></a>
        <h2>{html.escape(row['predicted'])} · {row['confidence']:.3f}</h2>
        <p>Target: <strong>{html.escape(row['target'])}</strong> · signed Grad-CAM from <code>{html.escape(row['layer'])}</code></p>
        <small>{html.escape(row['image'])}</small></article>"""
        for row in rows
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Trainer ConvNeXt xAI</title>
<style>body{{font:14px system-ui;background:#0d1117;color:#e6edf3;margin:28px}}header{{max-width:1100px;margin:auto}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px;max-width:1500px;margin:24px auto}}article{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}}img{{width:100%;background:#05070a}}h2{{font-size:17px}}p,small{{color:#9da7b3}}code{{color:#79c0ff}}</style></head><body><header><h1>Trainer classification xAI</h1>
<p>Architecture <code>{html.escape(architecture)}</code>. Red supports the selected class; blue opposes it. Activation tiles show the highest-energy channels at every captured feature stage.</p></header><main>{cards}</main></body></html>"""
    destination = output / "reports" / "xai.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf8")
    return destination


def generate_xai(model: nn.Module, payload: dict[str, Any], image_paths: Iterable[Path], output: Path,
                 device: torch.device, *, target: str = "predicted", max_images: int = 24,
                 intensity: float = 0.72, activation_channels: int = 4) -> dict[str, Any]:
    """Generate signed class evidence and stage activations for classifier images."""
    names = list(payload["classes"])
    paths = list(image_paths)[:max_images]
    if not paths:
        raise ValueError("xAI source contains no images")
    output = Path(output)
    xai_root = output / "xai"
    directories = {name: xai_root / name for name in ("inputs", "gradcam", "activations", "dossiers")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    layers = _feature_layers(model)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for layer_name, layer in layers:
        def capture(_module, _inputs, layer_output, *, name=layer_name):
            if isinstance(layer_output, torch.Tensor) and layer_output.ndim == 4:
                captured[name] = layer_output
        handles.append(layer.register_forward_hook(capture))

    rows: list[dict[str, Any]] = []
    model.eval()
    try:
        for index, image_path in enumerate(paths, start=1):
            dataset = ImageDataset([Sample(Path(image_path), 0)], int(payload["imgsz"]), augment=False,
                                   num_classes=len(names))
            tensor, _, _ = dataset[0]
            rgb = np.clip((tensor.numpy().transpose(1, 2, 0) * 0.25 + 0.5) * 255, 0, 255).astype(np.uint8)
            model_input = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            captured.clear()
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(tensor.unsqueeze(0).to(device))
                probabilities = logits.softmax(1)[0]
                target_index = _target_index(target, names, probabilities)
                if not captured:
                    raise ValueError("selected classifier feature layers did not produce spatial activations")
                final_name = next(reversed(captured))
                final_activation = captured[final_name]
                gradient = torch.autograd.grad(logits[0, target_index], final_activation, retain_graph=False)[0]
            weights = gradient.float().mean(dim=(2, 3), keepdim=True)
            signed = _normalise_signed((weights * final_activation.float()).sum(dim=1)[0].detach().cpu().numpy())
            overlay = _signed_overlay(model_input, signed, intensity)
            activations = _activation_montage(list(captured.items()), channels=activation_channels)
            probabilities_cpu = probabilities.detach().float().cpu()
            predicted_index = int(probabilities_cpu.argmax())
            token = _token(index, Path(image_path))

            display_side = max(320, int(payload["imgsz"]))
            display_input = _fit(model_input, display_side, display_side)
            display_overlay = _fit(overlay, display_side, display_side)
            input_panel = _label(display_input, "MODEL INPUT", f"{payload['imgsz']}px letterbox + checkpoint normalisation")
            overlay_panel = _label(
                display_overlay, f"SIGNED GRAD-CAM | {names[target_index]}",
                f"red supports | blue opposes | target p={float(probabilities_cpu[target_index]):.3f}",
            )
            top = np.concatenate([input_panel, overlay_panel], axis=1)
            activation_panel = _label(
                activations, "CONVNEXT FEATURE ACTIVATIONS",
                f"top {activation_channels} channels by mean absolute response at each stage",
            )
            activation_panel = _fit(activation_panel, top.shape[1], max(180, activation_panel.shape[0]))
            dossier = np.concatenate([top, activation_panel], axis=0)

            input_path = directories["inputs"] / f"{token}_input.jpg"
            signed_path = directories["gradcam"] / f"{token}_signed_gradcam.png"
            overlay_path = directories["gradcam"] / f"{token}_gradcam_overlay.jpg"
            activation_path = directories["activations"] / f"{token}_activations.jpg"
            dossier_path = directories["dossiers"] / f"{token}_dossier.jpg"
            cv2.imwrite(str(input_path), model_input)
            cv2.imwrite(str(signed_path), np.clip((signed + 1.0) * 127.5, 0, 255).astype(np.uint8))
            cv2.imwrite(str(overlay_path), overlay)
            cv2.imwrite(str(activation_path), activations)
            cv2.imwrite(str(dossier_path), dossier)

            top_values, top_indices = probabilities_cpu.topk(min(5, len(names)))
            rows.append({
                "token": token,
                "image": str(image_path),
                "architecture": str(payload.get("arch", "classifier")),
                "layer": final_name,
                "predicted": names[predicted_index],
                "confidence": float(probabilities_cpu[predicted_index]),
                "target": names[target_index],
                "target_score": float(probabilities_cpu[target_index]),
                "topk": [{"class": names[int(class_index)], "confidence": float(score)}
                         for score, class_index in zip(top_values, top_indices, strict=True)],
                "input": str(input_path.relative_to(output)),
                "signed_map": str(signed_path.relative_to(output)),
                "overlay": str(overlay_path.relative_to(output)),
                "activations": str(activation_path.relative_to(output)),
                "dossier": str(dossier_path.relative_to(output)),
            })
    finally:
        for handle in handles:
            handle.remove()

    manifest = {
        "format": "trainer-classification-xai-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": str(payload.get("arch", "classifier")),
        "classes": names,
        "method": "signed Grad-CAM; red supports selected class, blue opposes selected class",
        "images": rows,
    }
    gallery = _write_gallery(output, rows, str(payload.get("arch", "classifier")))
    manifest["gallery"] = str(gallery)
    (xai_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    with (xai_root / "predictions.csv").open("w", newline="", encoding="utf8") as handle:
        fields = ("image", "predicted", "confidence", "target", "target_score", "layer", "dossier")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    return manifest


__all__ = ["generate_xai"]
