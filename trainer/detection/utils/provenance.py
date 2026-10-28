"""Local, reproducible run identity; never contacts the Git remote."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from trainer.detection.utils.metrics import EVALUATOR_VERSION
from trainer.detection.utils.checks import cuda_device_index


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_provenance(run_dir: Path, cfg: dict, architecture: dict, loaders: dict, device) -> dict:
    """Collect before writing, so a rejected resume cannot overwrite its old provenance."""
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                                           stderr=subprocess.DEVNULL, timeout=10).decode()
        except (OSError, subprocess.SubprocessError):
            return ""

    patch = git("diff", "HEAD", "--", ".")
    paths = sorted(set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")) - {""})
    if not paths:  # installed package without a Git checkout
        paths = sorted(str(p.relative_to(root)) for p in (root / "trainer").rglob("*")
                       if p.is_file() and p.suffix in (".py", ".toml"))
    source_files = {p: file_sha256(root / p) for p in paths
                    if (root / p).is_file() and not (root / p).resolve().is_relative_to(run_dir.resolve())}
    source_hash = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    training_files = {p: digest for p, digest in source_files.items()
                      if p.startswith("trainer/") and Path(p).suffix in (".py", ".toml")}
    training_hash = hashlib.sha256(json.dumps(training_files, sort_keys=True).encode()).hexdigest()
    versions = {}
    for package in ("torch", "numpy", "opencv-python", "pycocotools", "torchvision"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    datasets = {}
    for split, loader in loaders.items():
        if loader is None:
            continue
        ds = loader.dataset
        digest = hashlib.sha256()
        for path, shape, labels in zip(ds.paths, ds.shapes, ds.labels):
            stat = Path(path).stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            digest.update(shape.tobytes())
            digest.update(labels.tobytes())
        identity = {"images": len(ds), "labels_shapes_paths_image_stats_sha256": digest.hexdigest(),
                    "image_identity": "paths, sizes and mtimes; image contents are not hashed"}
        if hasattr(ds, "annotations_json"):
            identity["annotations_sha256"] = file_sha256(ds.annotations_json)
        datasets[split] = identity
    result = {
        "git_sha": git("rev-parse", "HEAD").strip() or None,
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": source_hash, "source_files": source_files,
        "training_code_sha256": training_hash,
        "diff_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "tracked_diff": patch,
        "python": sys.version, "platform": platform.platform(), "versions": versions,
        "device": str(device), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(cuda_device_index(device)) if device.type == "cuda" else None,
        "seed": cfg["seed"], "deterministic": cfg["deterministic"], "architecture": architecture,
        "evaluator": EVALUATOR_VERSION, "datasets": datasets,
    }
    return result


def write_provenance(run_dir: Path, result: dict) -> None:
    (run_dir / "cfg").mkdir(parents=True, exist_ok=True)
    (run_dir / "cfg" / "source.patch").write_text(result["tracked_diff"], encoding="utf-8")
    write_json(run_dir / "cfg" / "provenance.json", result)
