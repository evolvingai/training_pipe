"""DetectionModel: build an nn.Module graph from a model TOML via the registry."""

from __future__ import annotations

import contextlib
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn

from trainer.detection.config import load_toml
from trainer.detection.config.loader import CFG_DIR
from trainer.detection.nn import modules as M  # noqa: F401  (populates the registry)
from trainer.detection.nn.modules.conv import Conv
from trainer.detection.nn.modules.heads import Detect
from trainer.detection.nn.registry import get_module
from trainer.detection.utils.logging import LOGGER

# modules whose first arg is output channels and that accept a repeat count n
_REPEATED = {"RELAN", "CSP", "A2Block", "C3k2", "A2C2f"}
# modules whose first arg is output channels
_CH_OUT = {"Conv", "DWConv", "SPPF"} | _REPEATED

SCALE_SUFFIXES = ("n", "s", "m", "l", "x")
SCALE_ALIASES = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}


def _split_model_scale(stem: str) -> tuple[str, str]:
    token = stem.rsplit("-", 1)[-1]
    scale = SCALE_ALIASES.get(token, token)
    if scale in SCALE_SUFFIXES and stem.endswith(f"-{token}"):
        return stem[: -(len(token) + 1)], scale
    return stem, "n"


def resolve_model_toml(name: str | Path) -> tuple[Path, str]:
    """Resolve 'trainer-n' / 'trainer-csp-s' / explicit path to (toml path, scale)."""
    p = Path(name)
    if p.suffix == ".toml":
        stem = p.stem
        base, scale = _split_model_scale(stem)
        if base != stem:
            candidate = p if p.exists() else p.with_name(f"{base}.toml")
            if candidate.exists():
                return candidate, scale
            candidate = CFG_DIR / "models" / f"{base}.toml"
            if candidate.exists():
                return candidate, scale
        if p.exists():
            return p, scale
        candidate = CFG_DIR / "models" / p.name
        if candidate.exists():
            return candidate, scale
        raise FileNotFoundError(f"model config not found: {name}")
    stem = str(name)
    base, scale = _split_model_scale(stem)
    for candidate_name in (stem, base):
        candidate = CFG_DIR / "models" / f"{candidate_name}.toml"
        if candidate.exists():
            return candidate, scale
    raise FileNotFoundError(f"model config not found: {name}")


def _make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor * math.ceil(value / divisor), divisor)


def parse_model(cfg: dict, ch_in: int = 3, scale: str = "n", nc: int | None = None,
                reg_max: int | None = None):
    """Build (nn.ModuleList, save_list, head_index) from a parsed model TOML dict."""
    nc = nc if nc is not None else int(cfg["nc"])
    if nc <= 0 or ch_in <= 0:
        raise ValueError("nc and input channels must be positive")
    scales = cfg.get("scales", {})
    if scales and scale not in scales:
        raise KeyError(f"unknown model scale '{scale}'; available: {sorted(scales)}")
    depth, width, max_ch = scales.get(scale, [1.0, 1.0, 1024])
    if float(depth) <= 0 or float(width) <= 0 or int(max_ch) <= 0:
        raise ValueError(f"invalid scale definition for '{scale}': {[depth, width, max_ch]}")
    reg_max = int(reg_max if reg_max is not None else cfg.get("reg_max", 16))
    if reg_max < 2:
        raise ValueError("reg_max must be at least 2")
    layers: list[nn.Module] = []
    save: set[int] = set()
    ch: list[int] = []  # ch[i] = output channels of layer i
    head_index = -1

    def ch_of(f: int) -> int:
        if f == -1:
            return ch[-1] if ch else ch_in
        return ch[f]

    def validate_from(frm: int | list[int], layer_index: int) -> None:
        refs = [frm] if isinstance(frm, int) else frm
        if not refs:
            raise ValueError(f"layer {layer_index}: 'from' cannot be empty")
        for ref in refs:
            if not isinstance(ref, int):
                raise TypeError(f"layer {layer_index}: invalid from index {ref!r}")
            if ref < -1 or ref >= layer_index:
                raise ValueError(f"layer {layer_index}: invalid from index {ref}")

    for i, (frm, repeats, mod_name, args) in enumerate(list(cfg["backbone"]) + list(cfg["head"])):
        validate_from(frm, i)
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError(f"layer {i}: repeats must be a positive integer")
        if not isinstance(args, list):
            raise TypeError(f"layer {i}: args must be a list")
        module_cls = get_module(mod_name)
        args = [nc if a == "nc" else reg_max if a == "reg_max" else a for a in args]
        n = max(round(repeats * depth), 1) if repeats > 1 else repeats

        if mod_name in _CH_OUT:
            if not isinstance(frm, int):
                raise TypeError(f"layer {i}: {mod_name} requires one input")
            c1 = ch_of(frm) if isinstance(frm, int) else ch_of(frm[0])
            c2 = _make_divisible(min(int(args[0]), int(max_ch)) * float(width))
            if mod_name == "C3k2" and scale in "mlx":
                args[1] = True
            if mod_name == "A2C2f" and scale in "lx":
                args.extend([True, 1.2])
            if mod_name in _REPEATED:
                module = module_cls(c1, c2, n, *args[1:])
            else:
                module = module_cls(c1, c2, *args[1:])
            out_ch = c2
        elif mod_name == "Concat":
            if not isinstance(frm, list):
                raise TypeError(f"layer {i}: Concat requires a list of inputs")
            module = module_cls(*args)
            out_ch = sum(ch_of(f) for f in frm)
        elif mod_name == "Upsample":
            module = module_cls(*args)
            out_ch = ch_of(frm) if isinstance(frm, int) else ch_of(frm[0])
        elif mod_name == "Detect":
            if not isinstance(frm, list):
                raise TypeError(f"layer {i}: Detect requires a list of feature inputs")
            if head_index >= 0:
                raise ValueError("model architecture has multiple Detect heads")
            in_chs = tuple(ch_of(f) for f in frm)
            module = module_cls(args[0], in_chs, *args[1:], reg_channels=int(cfg.get("reg_channels", 0)))
            out_ch = 0
            head_index = i
        else:
            module = module_cls(*args)
            out_ch = ch_of(frm) if isinstance(frm, int) else ch_of(frm[0])

        module.f = frm  # type: ignore[attr-defined]
        module.i = i  # type: ignore[attr-defined]
        layers.append(module)
        ch.append(out_ch)
        for f in ([frm] if isinstance(frm, int) else frm):
            if f != -1:
                save.add(f)  # absolute layer indices referenced later
    if head_index < 0:
        raise ValueError("model architecture has no Detect head")
    if head_index != len(layers) - 1:
        raise ValueError("Detect head must be the final model layer")
    return nn.ModuleList(layers), sorted(save), head_index


class DetectionModel(nn.Module):
    def __init__(self, model: str | Path | dict = "trainer-n", nc: int | None = None, ch: int = 3,
                 reg_max: int | None = None, reg_channels: int | None = None):
        super().__init__()
        if isinstance(model, dict):
            cfg = copy.deepcopy(model)
            scale = str(cfg.pop("scale", "n"))
            self.model_name = str(cfg.pop("name", "custom"))
        else:
            path, scale = resolve_model_toml(model)
            cfg = load_toml(path)
            base, _ = _split_model_scale(path.stem)
            self.model_name = f"{base}-{scale}"
        if reg_channels is not None:
            cfg["reg_channels"] = reg_channels
        self.cfg = cfg
        self.scale = scale
        self.ch = ch
        self.nc = nc if nc is not None else int(cfg["nc"])
        self.layers, self.save, self.head_index = parse_model(
            cfg, ch_in=ch, scale=scale, nc=self.nc, reg_max=reg_max,
        )

        # Infer strides without updating BatchNorm running statistics.
        dummy = torch.zeros(1, ch, 256, 256)
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            _, feats = self.forward(dummy)
        self.head.stride = torch.tensor([256 / f.shape[-2] for f in feats])
        self.head.bias_init()
        self.train(was_training)
        self.names: dict[int, str] = {i: str(i) for i in range(self.nc)}

    def architecture_spec(self) -> dict:
        """Return a portable architecture definition for checkpoints."""
        spec = copy.deepcopy(self.cfg)
        spec["scale"] = self.scale
        spec["name"] = self.model_name
        return spec

    @property
    def head(self) -> Detect:
        return self.layers[self.head_index]  # type: ignore[return-value]

    @property
    def stride(self) -> torch.Tensor:
        return self.head.stride

    def forward(self, x: torch.Tensor, return_features: bool = False):
        y: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            frm = layer.f
            if isinstance(frm, int):
                inp = x if frm == -1 else y[frm]
            else:
                inp = [x if f == -1 else y[f] for f in frm]
            if return_features and layer.i == self.head_index:
                return layer.forward_raw(inp), inp
            x = layer(inp)
            if layer.i in self.save:
                y[layer.i] = x
        return x

    def fuse(self) -> "DetectionModel":
        """Fold Conv+BN throughout for inference."""
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.fuse()
        return self

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def info(self) -> str:
        return (f"{self.model_name}: {len(self.layers)} layers, {self.num_params():,} params, "
                f"nc={self.nc}, strides={self.stride.tolist()}")

    def load_weights(self, state_dict: dict, strict: bool = False) -> None:
        """Load weights, silently skipping head tensors whose shape differs (nc remap)."""
        own = self.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        self.load_state_dict(filtered, strict=False)
        if skipped:
            LOGGER.info(f"transfer: loaded {len(filtered)}/{len(own)} tensors, "
                        f"skipped {len(skipped)} (shape mismatch / head remap)")
        elif strict and len(filtered) != len(own):
            missing = set(own) - set(filtered)
            raise RuntimeError(f"missing weights: {sorted(missing)[:5]}...")


with contextlib.suppress(Exception):  # nicer repr in logs
    DetectionModel.__repr__ = lambda self: self.info()  # type: ignore[method-assign]
