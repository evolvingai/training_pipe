"""Small offline VLM/LoRA architecture plus the production HF plugin marker."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from trainer.registry import register_architecture


class LoRALinear(nn.Module):
    def __init__(self, input_features: int, output_features: int, rank: int = 4, alpha: int = 8):
        super().__init__()
        self.base = nn.Linear(input_features, output_features)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, input_features))
        self.lora_b = nn.Parameter(torch.zeros(output_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + (value @ self.lora_a.t() @ self.lora_b.t()) * self.scale


class TinyVisionLanguageModel(nn.Module):
    """Image-conditioned autoregressive model intended for deterministic smoke tests."""

    def __init__(self, vocab_size: int = 64, embed_dim: int = 32, rank: int = 4):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.tokens = nn.Embedding(vocab_size, embed_dim)
        self.decoder = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.lm_head = LoRALinear(embed_dim, vocab_size, rank=rank, alpha=rank * 2)
        for module in (self.vision, self.tokens, self.decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        initial = self.vision(images).unsqueeze(0)
        sequence, _ = self.decoder(self.tokens(input_ids), initial)
        return self.lm_head(sequence)


@register_architecture("vlm", "tiny-vlm-lora", aliases=("tiny-vlm",),
                       description="Offline image-conditioned LM with trainable LoRA output adapters.",
                       family="Offline VLM",
                       input_contract="RGB images paired with token sequences.",
                       output_contract="Token logits with a portable LoRA adapter delta.",
                       training_strategy="Frozen vision/token/decoder modules with trainable low-rank output adapters.",
                       stages=("image + prompt", "vision encoder", "token decoder", "LoRA adapter", "token logits"))
def build_tiny_vlm(vocab_size: int = 64, embed_dim: int = 32, rank: int = 4, **_: object) -> nn.Module:
    return TinyVisionLanguageModel(vocab_size=vocab_size, embed_dim=embed_dim, rank=rank)


@register_architecture("vlm", "hf-lora",
                       description="Production AutoModelForImageTextToText + PEFT LoRA workflow.",
                       family="Hugging Face VLM",
                       input_contract="Image-text JSONL with image, prompt and response fields.",
                       output_contract="PEFT adapter weights, processor contract and optionally merged model.",
                       training_strategy="Parameter-efficient fine-tuning with frozen base weights and configurable LoRA targets.",
                       stages=("image + prompt", "processor", "frozen base VLM", "LoRA targets", "adapter"))
def build_hf_lora(**_: object):
    raise RuntimeError("hf-lora is constructed by the VLM task runner; provide model=<Hugging Face model id>")
