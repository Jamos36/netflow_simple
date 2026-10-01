from __future__ import annotations

from pathlib import Path

import yaml


DEFAULTS = {
    "data": {"paths": [], "timestamp_column": "flow_start_time", "timestamp_unit": None,
             "exclude_columns": [], "hash_features": 256, "batch_rows": 20_000, "train_sample_rows": 50_000},
    "split": {"train_fraction": 0.6, "validation_fraction": 0.2, "train_end": None, "validation_end": None},
    "context": {"pentest_start": None, "pentest_end": None, "pentest_months": []},
    "review": {"threshold_quantile": 0.995, "critical_quantile": 0.999, "high_quantile": 0.995,
               "medium_quantile": 0.990, "low_quantile": 0.975, "top_per_day": 100},
    "output": {"directory": "outputs"},
    "models": {"seed": 42, "iforest": {"n_estimators": 200, "max_samples": 256, "max_features": 1.0,
                                          "n_jobs": 4},
                "pca": {"max_components": 64, "explained_variance": 0.95},
                "hbos": {"bins": 20}, "gmm": {"components": 4, "max_iter": 50, "fit_rows": 10_000}},
}


def _merge(default: dict, supplied: dict) -> dict:
    result = dict(default)
    for key, value in supplied.items():
        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
            result[key] = _merge(default[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> tuple[dict, Path]:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    cfg = _merge(DEFAULTS, raw)
    base = path.parent
    cfg["data"]["paths"] = [str((base / p).resolve()) if not Path(p).is_absolute() else str(Path(p))
                              for p in cfg["data"]["paths"]]
    out = Path(cfg["output"]["directory"])
    cfg["output"]["directory"] = str((base / out).resolve() if not out.is_absolute() else out)
    for key in ("hash_features", "batch_rows", "train_sample_rows"):
        if int(cfg["data"][key]) < 1:
            raise ValueError(f"data.{key} must be positive")
    tr = float(cfg["split"]["train_fraction"])
    va = float(cfg["split"]["validation_fraction"])
    if tr <= 0 or va <= 0 or tr + va >= 1:
        raise ValueError("split fractions must be positive and sum to less than 1")
    if not 0.5 < float(cfg["review"]["threshold_quantile"]) < 1:
        raise ValueError("review.threshold_quantile must be between 0.5 and 1")
    band_quantiles = [float(cfg["review"][f"{name}_quantile"])
                      for name in ("low", "medium", "high", "critical")]
    if not 0.5 < band_quantiles[0] <= band_quantiles[1] <= band_quantiles[2] <= band_quantiles[3] < 1:
        raise ValueError("Review band quantiles must satisfy 0.5 < low <= medium <= high <= critical < 1")
    return cfg, base
