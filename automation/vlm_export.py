#!/usr/bin/env python3
"""Export a merged image-text VLM forward graph to ONNX or TensorRT.

The exported graph returns logits for one multimodal forward pass.  Production
autoregressive decoding remains the responsibility of the serving runtime.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType
from typing import Iterable


def _prune_unused_optional_outputs(path: Path) -> None:
    """Drop unused trailing ONNX outputs unsupported by TensorRT's parser."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False)
    consumed = {value for node in graph.graph.node for value in node.input}
    retained = {value.name for value in graph.graph.output}
    changed = False
    for node in graph.graph.node:
        while len(node.output) > 1 and node.output[-1] not in consumed and node.output[-1] not in retained:
            node.output.pop()
            changed = True
    if changed:
        onnx.save_model(graph, str(path))


def _simplify_for_tensorrt(path: Path) -> Path:
    """Constant-fold an ONNX copy so TensorRT receives literal reduce axes."""
    try:
        import onnxsim  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "VLM TensorRT export requires ONNX Simplifier; run: pip install -e '.[vlm,deploy]'"
        ) from error
    output = path.with_name(f"{path.stem}.tensorrt.onnx")
    output.unlink(missing_ok=True)
    subprocess.run(
        [sys.executable, "-m", "onnxsim", str(path), str(output), "--skip-optimization"],
        check=True,
    )
    return output


def _make_idefics3_exportable(model, torch) -> None:
    """Replace Idefics3's data-dependent positional-id loop with tensor ops.

    The upstream implementation iterates over a patch mask and calls arange
    with tensor-valued bounds.  That is valid eager PyTorch, but cannot be
    represented by torch.export/ONNX. The vectorized equivalent keeps patch
    padding exportable; the graph retains the fixed image shape of the trace.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type != "idefics3":
        return
    core = model.model
    vision = core.vision_model
    embeddings = vision.embeddings

    def export_forward(self, pixel_values, patch_attention_mask):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        output = patch_embeds.flatten(2).transpose(1, 2)
        patches_h = max_im_h // self.patch_size
        patches_w = max_im_w // self.patch_size
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side,
            device=pixel_values.device, dtype=pixel_values.dtype,
        )
        rows = torch.arange(patches_h, device=pixel_values.device, dtype=pixel_values.dtype)
        cols = torch.arange(patches_w, device=pixel_values.device, dtype=pixel_values.dtype)
        valid_h = patch_attention_mask[:, :, 0].sum(dim=1).clamp_min(1).to(pixel_values.dtype)
        valid_w = patch_attention_mask[:, 0, :].sum(dim=1).clamp_min(1).to(pixel_values.dtype)
        row_buckets = torch.bucketize(rows.unsqueeze(0) / valid_h.unsqueeze(1) * (1 - 1e-6), boundaries)
        col_buckets = torch.bucketize(cols.unsqueeze(0) / valid_w.unsqueeze(1) * (1 - 1e-6), boundaries)
        position_ids = (
            row_buckets.unsqueeze(2) * self.num_patches_per_side + col_buckets.unsqueeze(1)
        ).flatten(1)
        position_ids = torch.where(patch_attention_mask.flatten(1), position_ids, 0)
        return output + self.position_embedding(position_ids)

    embeddings.forward = MethodType(export_forward, embeddings)

    def vision_forward(self, pixel_values, patch_attention_mask=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutput

        if patch_attention_mask is None:
            patch_attention_mask = torch.ones(
                (pixel_values.shape[0], pixel_values.shape[2] // self.patch_size,
                 pixel_values.shape[3] // self.patch_size),
                dtype=torch.bool, device=pixel_values.device,
            )
        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
        flat_mask = patch_attention_mask.flatten(1)
        attention = (~flat_mask[:, None, None, :]).to(hidden_states.dtype) * torch.finfo(hidden_states.dtype).min
        encoded = self.encoder(inputs_embeds=hidden_states, attention_mask=attention).last_hidden_state
        return BaseModelOutput(last_hidden_state=self.post_layernorm(encoded))

    vision.forward = MethodType(vision_forward, vision)

    def image_features(self, pixel_values, pixel_attention_mask=None):
        batch_size, num_images, channels, height, width = pixel_values.shape
        pixels = pixel_values.reshape(batch_size * num_images, channels, height, width).to(dtype=self.dtype)
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                (batch_size, num_images, height, width), dtype=torch.bool, device=pixel_values.device,
            )
        masks = pixel_attention_mask.reshape(batch_size * num_images, 1, height, width)
        patch_size = self.config.vision_config.patch_size
        patch_mask = masks.reshape(
            batch_size * num_images, 1, height // patch_size, patch_size, width // patch_size, patch_size,
        ).any(dim=5).any(dim=3)
        states = self.vision_model(
            pixel_values=pixels, patch_attention_mask=patch_mask.squeeze(1),
        ).last_hidden_state
        return self.connector(states)

    core.get_image_features = MethodType(image_features, core)

def _dependencies():
    try:
        import torch
        from PIL import Image
        from peft import PeftModel
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM
    except ImportError as error:
        raise RuntimeError("VLM export support is not installed; run: pip install -e '.[vlm,deploy]'") from error
    return torch, Image, PeftModel, AutoProcessor, AutoVLM


def _device(torch, value: str):
    selected = value.strip().lower()
    if selected in {"", "auto"}:
        selected = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _dtype(torch, value: str, device):
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[value]


def _chat(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def _profile(shape: tuple[int, ...], dynamic_sequence: bool = False):
    minimum = list(shape); optimum = list(shape); maximum = list(shape)
    if shape:
        minimum[0] = optimum[0] = 1
        maximum[0] = max(1, min(4, shape[0] * 4))
    # Multimodal image-token counts are architecture-specific.  Keep sequence
    # length fixed to the sample contract even though ONNX names the axis.
    if dynamic_sequence and len(shape) > 1:
        minimum[1] = optimum[1] = maximum[1] = shape[1]
    return tuple(minimum), tuple(optimum), tuple(maximum)


def export(args: argparse.Namespace) -> int:
    torch, Image, PeftModel, AutoProcessor, AutoVLM = _dependencies()
    device = _device(torch, args.device)
    dtype = _dtype(torch, args.dtype, device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoVLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=args.trust_remote_code, attn_implementation="eager",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.config.use_cache = False
    model = model.eval().to(device)
    _make_idefics3_exportable(model, torch)

    image = Image.open(args.sample_image).convert("RGB") if args.sample_image else Image.new(
        "RGB", (args.image_size, args.image_size), color=(127, 127, 127),
    )
    try:
        text = processor.apply_chat_template(_chat(args.prompt), tokenize=False, add_generation_prompt=True)
        batch = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    finally:
        image.close()
    tensors = {key: value.to(device) for key, value in batch.items() if hasattr(value, "shape")}
    if "input_ids" in tensors and tensors["input_ids"].shape[1] > args.max_length:
        raise ValueError(
            f"processed prompt uses {tensors['input_ids'].shape[1]} tokens, above max-length={args.max_length}; "
            "increase max-length so image tokens are not truncated"
        )
    signature = inspect.signature(model.forward)
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if not accepts_kwargs:
        tensors = {key: value for key, value in tensors.items() if key in signature.parameters}
    if not tensors:
        raise RuntimeError("processor did not produce tensor inputs accepted by the VLM")
    supported_inputs = (
        "input_ids", "attention_mask", "pixel_values", "pixel_attention_mask", "image_sizes", "token_type_ids",
    )
    unknown = sorted(set(tensors) - set(supported_inputs))
    if unknown:
        raise RuntimeError(f"VLM processor produced unsupported export inputs: {', '.join(unknown)}")
    input_names = [name for name in supported_inputs if name in tensors]

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids=None, attention_mask=None, pixel_values=None, pixel_attention_mask=None,
                    image_sizes=None, token_type_ids=None):
            values = (input_ids, attention_mask, pixel_values, pixel_attention_mask, image_sizes, token_type_ids)
            kwargs = {name: value for name, value in zip(supported_inputs, values, strict=True) if value is not None}
            if getattr(self.wrapped.config, "model_type", "") == "idefics3":
                core = self.wrapped.model
                inputs_embeds = core.text_model.get_input_embeddings()(input_ids)
                image_states = core.get_image_features(pixel_values, pixel_attention_mask)
                inputs_embeds = core.inputs_merger(input_ids, inputs_embeds, image_states)
                sequence = inputs_embeds.shape[1]
                minimum = torch.finfo(inputs_embeds.dtype).min
                causal = torch.triu(
                    torch.full((sequence, sequence), minimum, dtype=inputs_embeds.dtype,
                               device=inputs_embeds.device),
                    diagonal=1,
                )[None, None, :, :]
                if attention_mask is not None:
                    causal = causal.masked_fill(attention_mask[:, None, None, :] == 0, minimum)
                hidden = core.text_model(
                    inputs_embeds=inputs_embeds, attention_mask=causal, use_cache=False, return_dict=True,
                ).last_hidden_state
                return self.wrapped.lm_head(hidden)
            return self.wrapped(**kwargs, use_cache=False).logits

    output = Path(args.output)
    stem = output.with_suffix("") if output.suffix.lower() in {".onnx", ".engine"} else output
    onnx_output = stem.with_suffix(".onnx")
    onnx_output.parent.mkdir(parents=True, exist_ok=True)
    values = tuple(tensors.get(name) for name in supported_inputs)
    with torch.inference_mode():
        torch.onnx.export(
            LogitsWrapper(model), values, onnx_output, input_names=input_names,
            output_names=["logits"], opset_version=args.opset, dynamo=True,
        )
    _prune_unused_optional_outputs(onnx_output)

    profiles = {
        name: _profile(
            tuple(int(value) for value in tensors[name].shape),
            name in {"input_ids", "attention_mask", "token_type_ids"},
        )
        for name in input_names
    }
    contract = {
        "format": "trainer-vlm-onnx-v1", "base_model": args.model, "adapter": args.adapter or None,
        "onnx": str(onnx_output), "inputs": {
            name: {"shape": list(tensors[name].shape), "dtype": str(tensors[name].dtype).replace("torch.", "")}
            for name in input_names
        }, "output": "logits", "prompt": args.prompt,
        "runtime": "fixed-shape forward logits; autoregressive decoding is handled by the serving runtime",
    }
    final_output = onnx_output
    if args.format != "onnx":
        from trainer.detection.engine.exporter import export_tensorrt

        tensorrt_onnx = _simplify_for_tensorrt(onnx_output)
        try:
            final_output = export_tensorrt(
                tensorrt_onnx, half=args.half, workspace_gb=args.workspace,
                out_path=stem.with_suffix(".engine"), profile_shapes=profiles,
            )
        finally:
            tensorrt_onnx.unlink(missing_ok=True)
            Path(f"{tensorrt_onnx}.data").unlink(missing_ok=True)
        contract.update({
            "format": "trainer-vlm-tensorrt-v1", "engine": str(final_output),
            "precision": str(dtype).replace("torch.", ""),
        })
    stem.with_suffix(".json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf8")
    processor.save_pretrained(stem.parent / "processor")
    print(f"exported VLM: {final_output}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--model", required=True)
    root.add_argument("--adapter", default="")
    root.add_argument("--output", required=True)
    root.add_argument("--format", choices=["onnx", "tensorrt"], default="onnx")
    root.add_argument("--device", default="auto")
    root.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    root.add_argument("--sample-image", default="")
    root.add_argument("--prompt", default="Describe the image.")
    root.add_argument("--image-size", type=int, default=384)
    root.add_argument("--max-length", type=int, default=4096)
    root.add_argument("--opset", type=int, default=18)
    root.add_argument("--workspace", type=int, default=8)
    root.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    root.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return root


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.format == "tensorrt" and not args.device.lower().startswith("cuda"):
        raise ValueError("TensorRT VLM export requires a CUDA device")
    return export(args)


if __name__ == "__main__":
    raise SystemExit(main())
