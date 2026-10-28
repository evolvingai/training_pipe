"""MuSGD: Muon-orthogonalized updates on weight matrices, added to SGD momentum.

Muon (Jordan et al., 2024) replaces a weight matrix's raw momentum with its nearest
orthogonal matrix, computed by a quintic Newton-Schulz iteration. Orthogonalizing
equalizes the update's singular values, so no single direction dominates a step. It
only makes sense for parameters that *are* matrices; 1-D parameters (biases,
BatchNorm affine, scalar gains) get plain SGD.

MuSGD runs both halves on matrix parameters and sums them, which keeps SGD's
well-understood behaviour underneath while adding Muon's conditioning on top. Groups
opt in with `use_muon`; the blend is set by `muon_weight` / `sgd_weight`.

Written from the published algorithm, not ported from any implementation.
"""

from __future__ import annotations

import time

import torch
from torch.optim import Optimizer

from trainer.detection.utils.checks import cuda_device_index

# Quintic coefficients tuned to maximize convergence slope at zero. The iteration
# X <- a*X + (b*A + c*A@A)@X with A = X@X.T drives singular values toward ~1 without
# converging exactly -- they land in roughly [0.5, 1.5], which is good enough for an
# optimizer and far cheaper than an SVD.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize a 2-D matrix via Newton-Schulz iteration.

    Returns a tensor shaped like `matrix` whose singular values are all near 1.
    Runs in bfloat16 on CUDA (the iteration is robust to it and this is the bulk of
    the optimizer's cost) and float32 elsewhere. An all-zero input returns all zeros
    rather than NaN.
    """
    if matrix.ndim != 2:
        raise ValueError(f"newton_schulz expects a 2-D matrix, got shape {tuple(matrix.shape)}")
    if steps < 1:
        raise ValueError("Newton-Schulz steps must be positive")
    use_bf16 = matrix.is_cuda and torch.cuda.get_device_capability(cuda_device_index(matrix.device))[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    # Normalize so the top singular value is <= 1, the iteration's basin of
    # convergence. Reduce in fp32: a conv weight flattens to ~10^5 entries and a
    # bfloat16 sum of squares over that many terms loses the small ones entirely.
    x = matrix.float()  # no-op for the fp32 gradients the trainer actually passes
    x = (x / (x.norm() + eps)).to(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a = x @ x.T
        b = _NS_B * a + _NS_C * (a @ a)
        x = _NS_A * x + b @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


def batched_newton_schulz(matrices: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Equal-shaped matrices; per-matrix normalization, never a global batch norm."""
    if matrices.ndim != 3 or steps < 1:
        raise ValueError("expected (batch, rows, cols) matrices and positive NS steps")
    bf16 = matrices.is_cuda and torch.cuda.get_device_capability(cuda_device_index(matrices.device))[0] >= 8
    x = matrices.float()
    x = (x / (x.norm(dim=(-2, -1), keepdim=True) + eps)).to(torch.bfloat16 if bf16 else torch.float32)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        a = x @ x.transpose(-2, -1)
        x = _NS_A * x + (_NS_B * a + _NS_C * (a @ a)) @ x
    return (x.transpose(-2, -1) if transposed else x).to(matrices.dtype)


def orthogonalize_updates(updates: list[torch.Tensor], splits: int, steps: int, batched: bool):
    """Q,K,V occupy contiguous output-channel thirds in an attention qkv convolution."""
    matrices, owners = [], []
    for index, update in enumerate(updates):
        if update.shape[0] % splits:
            raise ValueError("QKV output channels must divide into equal Q/K/V slices")
        for part in update.chunk(splits, dim=0):
            matrices.append(part.reshape(part.shape[0], -1))
            owners.append(index)
    results = [None] * len(matrices)
    if batched:
        buckets = {}
        for index, matrix in enumerate(matrices):
            buckets.setdefault((matrix.shape, matrix.device, matrix.dtype), []).append(index)
        for indices in buckets.values():
            for start in range(0, len(indices), 32):  # bound temporary batch workspace
                ids = indices[start:start + 32]
                out = batched_newton_schulz(torch.stack([matrices[i] for i in ids]), steps=steps)
                for i, item in zip(ids, out):
                    results[i] = item
    else:
        results = [newton_schulz(matrix, steps=steps) for matrix in matrices]
    pieces = [[] for _ in updates]
    for index, matrix, result in zip(owners, matrices, results):
        scale = max(1.0, matrix.shape[0] / matrix.shape[1]) ** .5
        pieces[index].append(result * scale)
    return [torch.cat(parts, dim=0).reshape(update.shape) for parts, update in zip(pieces, updates)]


def muon_weight_at_epoch(cfg: dict, epoch: int) -> float:
    """Epoch index is zero-based. The final taper epoch uses exactly zero weight."""
    weight = float(cfg["muon_weight"])
    if cfg.get("muon_schedule", "constant") == "constant":
        return weight
    start, end = int(cfg["muon_taper_start"]), int(cfg["muon_taper_end"])
    return weight * min(max((end - 1 - epoch) / (end - 1 - start), 0.), 1.)


class MuSGD(Optimizer):
    """SGD with momentum, plus an orthogonalized Muon update on `use_muon` groups.

    Per matrix parameter the two halves are summed:
        p -= lr * muon_weight * orthogonalize(muon_momentum) * max(1, rows/cols)**0.5
        p -= lr * sgd_weight  * sgd_momentum
    Groups with `use_muon=False` get only the second line, at the full `lr`. The
    aspect-ratio factor keeps the orthogonalized update's norm comparable across
    weights of different shape.

    The two momentum buffers deliberately use different conventions -- Muon's is an
    exponential moving average (`beta*buf + (1-beta)*grad`), SGD's is PyTorch's
    undamped accumulator (`momentum*buf + grad`). Both are their own algorithm's
    convention. `muon_beta=-1` preserves coupled momentum; an explicit beta
    controls only the Muon EMA, independent of SGD warmup/momentum.

    Weight decay is applied once, to the SGD half only.

    `muon_weight` and `sgd_weight` live in the param groups rather than on the
    instance so that `state_dict()` carries them and a resumed run restores the same
    blend it was trained with.
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False, use_muon: bool = False,
                 muon_weight: float = 0.2, sgd_weight: float = 1.0, muon_beta: float = -1,
                 ns_steps: int = 5, batch_ns: bool = False):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if muon_weight < 0.0 or sgd_weight < 0.0:
            raise ValueError(f"invalid blend weights: muon={muon_weight}, sgd={sgd_weight}")
        if not (muon_beta == -1 or 0 <= muon_beta < 1) or ns_steps < 1:
            raise ValueError("invalid Muon beta or NS steps")
        super().__init__(params, {
            "lr": lr, "momentum": momentum, "weight_decay": weight_decay,
            "nesterov": nesterov, "use_muon": use_muon,
            "muon_weight": muon_weight, "sgd_weight": sgd_weight,
            "muon_beta": muon_beta, "ns_steps": ns_steps, "batch_ns": batch_ns, "muon_splits": 1,
        })
        self.collect_diagnostics = False
        self.profile = False
        self.last_diagnostics = []
        self.ns_seconds = 0.

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.last_diagnostics = []
        self.ns_seconds = 0.
        for group_index, group in enumerate(self.param_groups):
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            use_muon = group["use_muon"]
            sgd_lr = lr * group["sgd_weight"] if use_muon else lr
            beta = group.get("muon_beta", -1)
            beta = momentum if beta == -1 else beta
            active_muon = use_muon and group["muon_weight"] != 0
            params, muon_updates = [], []
            for index, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                params.append((index, p))
                grad = p.grad
                state = self.state[p]
                # Checked independently: a run resumed with use_muon newly enabled has
                # sgd_buf already but no muon_buf.
                if "sgd_buf" not in state:
                    state["sgd_buf"] = torch.zeros_like(p)
                if active_muon and "muon_buf" not in state:
                    state["muon_buf"] = torch.zeros_like(p)

                if active_muon:
                    buf = state["muon_buf"]
                    buf.mul_(beta).add_(grad, alpha=1.0 - beta)
                    muon_updates.append(buf.mul(beta).add_(grad, alpha=1.0 - beta) if nesterov else buf)
            orthogonal = []
            if active_muon and params:
                device = params[0][1].device
                if self.profile and device.type == "cuda":
                    torch.cuda.synchronize(cuda_device_index(device))
                started = time.perf_counter()
                orthogonal = orthogonalize_updates(muon_updates, group.get("muon_splits", 1),
                                                   group.get("ns_steps", 5), group.get("batch_ns", False))
                if self.profile and device.type == "cuda":
                    torch.cuda.synchronize(cuda_device_index(device))
                self.ns_seconds += time.perf_counter() - started
            for position, (index, p) in enumerate(params):
                grad, state = p.grad, self.state[p]
                # Both directions (including decay) see the pre-update parameter.
                sgd_grad = grad.add(p, alpha=weight_decay) if weight_decay else grad
                buf = state["sgd_buf"]
                buf.mul_(momentum).add_(sgd_grad)
                update = sgd_grad.add(buf, alpha=momentum) if nesterov else buf
                if self.collect_diagnostics:
                    sg = update.float() * sgd_lr
                    mu = orthogonal[position].float() * (lr * group["muon_weight"]) if active_muon else torch.zeros_like(sg)
                    sn, mn, pn = sg.norm(), mu.norm(), p.float().norm()
                    denom = sn * mn
                    self.last_diagnostics.append({
                        "parameter": group.get("param_names", [str(i) for i in range(len(group["params"]))])[index],
                        "group": group_index, "parameter_norm": float(pn), "sgd_update_norm": float(sn),
                        "muon_update_norm": float(mn), "combined_update_norm": float((sg + mu).norm()),
                        "cosine": float((sg * mu).sum() / denom) if float(denom) else None,
                        "lr": lr, "muon_weight": group["muon_weight"], "muon_beta": beta,
                    })
                if active_muon:
                    p.add_(orthogonal[position], alpha=-(lr * group["muon_weight"]))
                p.add_(update, alpha=-sgd_lr)

        return loss
