from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _metric_values(metrics_per_fold: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in metrics_per_fold:
        value = item.get("metrics", {}).get(key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            values.append(float(value))
    return values


def aggregate_folds(result_dirs: list[Path]) -> dict:
    """Aggregate fold-level metrics into summary statistics."""

    metrics_per_fold = []
    runtime = 0.0
    for result_dir in result_dirs:
        payload = json.loads((result_dir / "metrics.json").read_text())
        metrics_per_fold.append(payload)
        runtime += float(payload.get("runtime_seconds", 0.0))

    keys: set[str] = set()
    for m in metrics_per_fold:
        keys.update(m.get("metrics", {}).keys())

    summary = {}
    for key in sorted(keys):
        vals = _metric_values(metrics_per_fold, key)
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    attn: dict[str, float] = {}
    for m in metrics_per_fold:
        for k, v in m.get("attention_weights_mean", {}).items():
            attn.setdefault(k, 0.0)
            attn[k] += float(v)
    if attn:
        for k in attn:
            attn[k] /= len(metrics_per_fold)

    return {
        "n_folds": len(metrics_per_fold),
        "metrics_per_fold": metrics_per_fold,
        "metrics_summary": summary,
        "attention_weights_mean": attn,
        "total_runtime_seconds": runtime,
    }
