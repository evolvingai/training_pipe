#!/usr/bin/env python3
"""LoRA tuning, evaluation, and adapter merge workflow for image-text VLMs.

The optional ``vlm`` dependency group is imported only when a tuning, evaluation, or merge
job runs, so detection and classification agents stay lightweight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from trainer.augmentation import (
    PASSTHROUGH_AUGMENTATION_DEFAULTS,
    add_augmentation_arguments,
    augmentation_config as resolve_augmentation_config,
    augmentation_overrides,
    build_pipeline,
)


@dataclass(frozen=True)
class VLMRecord:
    image: Path
    prompt: str
    answer: str


def _message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")).strip() for item in content
                         if isinstance(item, dict) and item.get("type") == "text").strip()
    return ""


def _message_image(message: dict) -> str:
    content = message.get("content", [])
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                return str(item.get("path") or item.get("image") or item.get("url") or "").strip()
    return ""


def parse_record(value: dict, base_dir: Path, line_number: int = 0) -> VLMRecord:
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: each JSONL row must be an object")
    image = str(value.get("image") or "").strip()
    prompt = str(value.get("prompt") or "").strip()
    answer = str(value.get("answer") or "").strip()
    messages = value.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            raise ValueError(f"line {line_number}: messages must be a list")
        user = next((item for item in messages if isinstance(item, dict) and item.get("role") == "user"), {})
        assistant = next((item for item in reversed(messages)
                          if isinstance(item, dict) and item.get("role") == "assistant"), {})
        image = image or _message_image(user)
        prompt = prompt or _message_text(user)
        answer = answer or _message_text(assistant)
    if not image or not prompt or not answer:
        raise ValueError(f"line {line_number}: image, prompt, and answer are required")
    if image.startswith(("http://", "https://")):
        raise ValueError(f"line {line_number}: remote images are not allowed; stage the image on the training VM")
    image_path = Path(image)
    if not image_path.is_absolute():
        image_path = (base_dir / image_path).resolve()
    return VLMRecord(image=image_path, prompt=prompt, answer=answer)


def load_records(path: Path, max_samples: int = 0) -> list[VLMRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"VLM JSONL dataset does not exist: {path}")
    records = []
    with path.open(encoding="utf8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(parse_record(json.loads(line), path.parent, line_number))
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
            if max_samples and len(records) >= max_samples:
                break
    if not records:
        raise ValueError("VLM dataset is empty")
    missing = [str(record.image) for record in records if not record.image.is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"{len(missing)} dataset image(s) are missing: {preview}")
    return records


def require_vlm_dependencies():
    try:
        import torch
        from PIL import Image
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM
    except ImportError as error:
        raise RuntimeError("VLM support is not installed; run: pip install -e '.[vlm]'") from error
    return torch, Image, LoraConfig, PeftModel, get_peft_model, AutoProcessor, AutoVLM


def select_device(torch, value: str):
    value = value.strip().lower()
    if value in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def dtype_for(torch, value: str, device):
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if device.type == "cuda" else torch.float32)
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[value]


def conversation(record: VLMRecord, include_answer: bool) -> list[dict]:
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": record.prompt},
    ]}]
    if include_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": record.answer}]})
    return messages


class VLMCollator:
    def __init__(self, processor, image_class, torch, max_length: int, transforms=None):
        self.processor, self.image_class, self.torch, self.max_length = processor, image_class, torch, max_length
        self.transforms = transforms

    def __call__(self, records: list[VLMRecord]) -> dict:
        images = [self.image_class.open(record.image).convert("RGB") for record in records]
        if self.transforms is not None:
            augmented = []
            for image in images:
                bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                sample = self.transforms({"image": bgr, "boxes": np.zeros((0, 4), np.float32),
                                          "cls": np.zeros((0,), np.float32)})
                augmented.append(self.image_class.fromarray(cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB)))
                image.close()
            images = augmented
        full_texts = [self.processor.apply_chat_template(conversation(record, True), tokenize=False,
                                                          add_generation_prompt=False) for record in records]
        prompt_texts = [self.processor.apply_chat_template(conversation(record, False), tokenize=False,
                                                            add_generation_prompt=True) for record in records]
        batch = self.processor(text=full_texts, images=images, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        prompt_batch = self.processor(text=prompt_texts, images=images, padding=True, truncation=True,
                                      max_length=self.max_length, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(1).tolist()
        for index, length in enumerate(prompt_lengths):
            labels[index, :min(int(length), labels.shape[1])] = -100
        batch["labels"] = labels
        for image in images:
            image.close()
        return batch


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def tune(args: argparse.Namespace) -> int:
    torch, Image, LoraConfig, _, get_peft_model, AutoProcessor, AutoVLM = require_vlm_dependencies()
    from torch.utils.data import DataLoader

    random.seed(args.seed); torch.manual_seed(args.seed)
    records = load_records(Path(args.data), args.max_samples)
    device = select_device(torch, args.device); dtype = dtype_for(torch, args.dtype, device)
    print(f"loading VLM {args.model} on {device} with {dtype}; {len(records)} examples")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoVLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    targets = args.target_modules if args.target_modules == "all-linear" else [
        item.strip() for item in args.target_modules.split(",") if item.strip()
    ]
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora).to(device)
    model.print_trainable_parameters()
    augmentation = resolve_augmentation_config(
        augmentation_overrides(args), defaults=PASSTHROUGH_AUGMENTATION_DEFAULTS,
        config_path=args.augmentation_config,
    )
    augmentation["imgsz"] = args.imgsz
    transforms = build_pipeline(augmentation, train=True, seed=args.seed, allow_multisample=False)
    collator = VLMCollator(processor, Image, torch, args.max_length, transforms)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(records, batch_size=args.batch, shuffle=True, collate_fn=collator, generator=generator)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history, updates = [], 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train(); running, batches = 0.0, 0
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            amp_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                loss = model(**batch).loss / args.gradient_accumulation
            loss.backward(); running += float(loss.detach()) * args.gradient_accumulation; batches += 1
            if step % args.gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); updates += 1
            if step % args.log_every == 0 or step == len(loader):
                print(f"epoch {epoch + 1}/{args.epochs} step {step}/{len(loader)} loss={running / batches:.4f}")
        history.append({"epoch": epoch + 1, "loss": running / max(batches, 1), "optimizer_updates": updates})
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    processor.save_pretrained(adapter)
    save_json(output / "metrics.json", {"model": args.model, "samples": len(records), "history": history})
    save_json(output / "augmentation.json", augmentation)
    save_json(output / "run.json", {"format": "trainer-vlm-lora-v1", "base_model": args.model,
              "adapter": str(adapter), "target_modules": targets, "lora_r": args.lora_r,
              "lora_alpha": args.lora_alpha, "device": str(device)})
    print(f"VLM adapter artifacts: {adapter}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    torch, Image, _, PeftModel, _, AutoProcessor, AutoVLM = require_vlm_dependencies()
    from torch.utils.data import DataLoader

    records = load_records(Path(args.data), args.max_samples)
    device = select_device(torch, args.device); dtype = dtype_for(torch, args.dtype, device)
    print(f"loading VLM {args.model} on {device} with {dtype}; evaluating {len(records)} examples")
    processor_source = args.adapter or args.model
    try:
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=args.trust_remote_code)
    except (OSError, ValueError):
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoVLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code)
    if args.adapter:
        adapter = Path(args.adapter)
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"PEFT adapter_config.json is missing under {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device); model.eval()
    collator = VLMCollator(processor, Image, torch, args.max_length)
    loader = DataLoader(records, batch_size=args.batch, shuffle=False, collate_fn=collator)
    loss_sum, evaluated_tokens, correct_tokens = 0.0, 0, 0
    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            amp_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                output = model(**batch)
            labels = batch["labels"][:, 1:]
            predictions = output.logits[:, :-1].argmax(-1)
            mask = labels.ne(-100)
            tokens = int(mask.sum().item())
            evaluated_tokens += tokens
            correct_tokens += int((predictions.eq(labels) & mask).sum().item())
            loss_sum += float(output.loss.detach()) * tokens
            print(f"evaluation batch {step}/{len(loader)} loss={float(output.loss.detach()):.4f}")
    if not evaluated_tokens:
        raise ValueError("VLM evaluation produced no answer tokens; increase max-length or check the answers")
    loss = loss_sum / evaluated_tokens
    metrics = {
        "model": args.model, "adapter": args.adapter or None, "samples": len(records),
        "evaluated_tokens": evaluated_tokens, "loss": loss,
        "perplexity": math.exp(min(loss, 30.0)),
        "token_accuracy": correct_tokens / max(evaluated_tokens, 1),
    }
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    save_json(output / "metrics.json", metrics)
    print(f"VLM evaluation: loss={loss:.4f} token_accuracy={metrics['token_accuracy']:.4f}")
    return 0


def merge(args: argparse.Namespace) -> int:
    torch, _, _, PeftModel, _, AutoProcessor, AutoVLM = require_vlm_dependencies()
    device = select_device(torch, args.device); dtype = dtype_for(torch, args.dtype, device)
    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter_config.json is missing under {adapter}")
    model = AutoVLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).save_pretrained(output)
    save_json(output / "trainer_merge.json", {"format": "trainer-vlm-merged-v1", "base_model": args.model,
                                               "adapter": str(adapter)})
    print(f"merged VLM artifacts: {output}")
    return 0


def common_model_options(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument("--model", required=True)
    command_parser.add_argument("--device", default="auto")
    command_parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    command_parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Trainer VLM LoRA tuning, evaluation, and adapter merge")
    sub = root.add_subparsers(dest="command", required=True)
    tune_parser = sub.add_parser("tune"); common_model_options(tune_parser)
    tune_parser.add_argument("--data", required=True); tune_parser.add_argument("--output", required=True)
    tune_parser.add_argument("--epochs", type=int, default=1); tune_parser.add_argument("--batch", type=int, default=1)
    tune_parser.add_argument("--lr", type=float, default=2e-5); tune_parser.add_argument("--weight-decay", type=float, default=0.0)
    tune_parser.add_argument("--max-length", type=int, default=1024); tune_parser.add_argument("--max-samples", type=int, default=0)
    tune_parser.add_argument("--imgsz", type=int, default=0)
    tune_parser.add_argument("--lora-r", type=int, default=16); tune_parser.add_argument("--lora-alpha", type=int, default=32)
    tune_parser.add_argument("--lora-dropout", type=float, default=0.05)
    tune_parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    tune_parser.add_argument("--gradient-accumulation", type=int, default=8)
    tune_parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    tune_parser.add_argument("--max-grad-norm", type=float, default=1.0); tune_parser.add_argument("--log-every", type=int, default=10)
    tune_parser.add_argument("--seed", type=int, default=0)
    add_augmentation_arguments(tune_parser)
    evaluate_parser = sub.add_parser("evaluate"); common_model_options(evaluate_parser)
    evaluate_parser.add_argument("--data", required=True); evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--adapter", default=""); evaluate_parser.add_argument("--batch", type=int, default=1)
    evaluate_parser.add_argument("--max-length", type=int, default=1024)
    evaluate_parser.add_argument("--max-samples", type=int, default=1000)
    merge_parser = sub.add_parser("merge"); common_model_options(merge_parser)
    merge_parser.add_argument("--adapter", required=True); merge_parser.add_argument("--output", required=True)
    return root


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for key in ("epochs", "batch", "max_length", "lora_r", "lora_alpha", "gradient_accumulation", "log_every"):
        if hasattr(args, key) and getattr(args, key) < 1:
            raise ValueError(f"{key.replace('_', '-')} must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "tune":
        return tune(args)
    if args.command == "evaluate":
        return evaluate(args)
    return merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
