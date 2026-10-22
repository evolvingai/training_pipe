"""Exponential moving average of model weights with warmup-aware decay ramp."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    """EMA whose decay ramps 0 -> decay_max over ~warmup_steps optimizer steps,
    so the average tracks the model closely early and stabilizes late."""

    def __init__(self, model: nn.Module, decay_max: float = 0.9999, warmup_steps: int = 2000):
        self.ema = copy.deepcopy(self._unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay_max = decay_max
        self.warmup_steps = warmup_steps
        self.step_count = 0

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return getattr(model, "_orig_mod", model)

    def _decay(self) -> float:
        return self.decay_max * (1 - math.exp(-self.step_count / max(self.warmup_steps, 1)))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step_count += 1
        d = self._decay()
        msd = self._unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.dtype), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> dict:
        return {
            "ema": self.ema.state_dict(),
            "step_count": self.step_count,
            "decay_max": self.decay_max,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema.load_state_dict(state["ema"])
        self.step_count = state["step_count"]
        self.decay_max = state["decay_max"]
        self.warmup_steps = state["warmup_steps"]
