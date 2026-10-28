"""Explicit exact-resume versus schedule-continuation configuration."""

from trainer.detection.config.loader import load_config, merge_overrides
from trainer.detection.data.epoch_sampler import DATA_RNG_VERSION

RUNTIME_OVERRIDES = {
    "resume", "resume_mode", "device", "workers", "project", "name", "exist_ok",
    "save_period", "mlflow", "mlflow_tracking_uri", "mlflow_experiment", "mlflow_run_name",
    "tracker_retries", "tracker_backoff", "log_per_class",
}
CONTINUATION_OVERRIDES = {"epochs", "lr0", "lrf", "cos_lr", "label_smoothing"}
BRANCH_OVERRIDES = {"muon_schedule", "muon_taper_start", "muon_taper_end", "profile", "muon_log_interval",
                    "loss_diagnostics"}


def resume_config(checkpoint: dict, overrides: dict) -> dict:
    saved = dict(checkpoint["cfg"])
    if saved.get("optimizer") == "auto" and "optimizer" in checkpoint:
        group = checkpoint["optimizer"]["param_groups"][0]
        saved["optimizer"] = "adamw" if "betas" in group else ("musgd" if "muon_weight" in group else "sgd")
    cfg = merge_overrides(load_config(), saved)  # reject unsupported legacy keys explicitly
    merged = merge_overrides(cfg, overrides)
    mode = merged["resume_mode"]
    allowed = RUNTIME_OVERRIDES | (CONTINUATION_OVERRIDES if mode == "continue" else
                                   BRANCH_OVERRIDES if mode == "branch" else set())
    changed = [k for k in overrides if k not in allowed and cfg[k] != merged[k]]
    if changed:
        raise ValueError(f"incompatible {mode} resume overrides: {sorted(changed)}; "
                         "use resume_mode='continue' for schedule changes or finetune for a new recipe")
    if mode in ("exact", "branch") and checkpoint.get("data_rng_version") != DATA_RNG_VERSION:
        raise ValueError("legacy checkpoint lacks epoch-addressed data RNG state; "
                         "use resume_mode='continue' (not an exact replay)")
    if "optimizer" not in checkpoint or "raw_model" not in checkpoint:
        raise ValueError("resume requires a training checkpoint (last.pt or epoch_N.pt), not inference-only weights")
    if mode == "continue" and int(merged["epochs"]) <= int(checkpoint["epoch"]) + 1:
        raise ValueError("continuation epochs must exceed the completed epoch count (epochs is the total horizon)")
    if mode == "branch" and int(merged["epochs"]) <= int(checkpoint["epoch"]) + 1:
        raise ValueError("branch requires an unfinished training horizon")
    return load_config(merged)
