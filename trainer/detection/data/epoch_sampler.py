"""Epoch-addressed sample randomness, independent of worker scheduling/prefetch."""

from __future__ import annotations

import hashlib

from torch.utils.data import Sampler

DATA_RNG_VERSION = "epoch-draw-v1"


class EpochSampler(Sampler):
    """Attach a seed and augmentation phase to every sampled index.

    Draw position distinguishes repeated samples under weighted sampling. Sending
    the phase in the index also reaches persistent workers without mutable shared
    dataset state or rebuilding the loader. Resume is supported at epoch boundaries.
    """

    def __init__(self, sampler, seed: int, rank: int, epochs: int, close_mosaic: int):
        self.sampler = sampler
        self.seed = seed
        self.rank = rank
        self.epoch = 0
        self.close_at = max(epochs - close_mosaic, 0) if close_mosaic > 0 else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        closed = self.close_at is not None and self.epoch >= self.close_at
        for draw, index in enumerate(self.sampler):
            key = f"{self.seed}:{self.epoch}:{self.rank}:{draw}:{index}".encode()
            seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
            yield int(index), seed, closed

    def __len__(self):
        return len(self.sampler)
