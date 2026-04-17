from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from scatterrad.config import TargetType, load_case_targets, load_targets_schema
from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.splits import _strat_value

log = logging.getLogger(__name__)


def _target_basenames(targets_tr: Path) -> list[str]:
    return sorted(p.stem for p in targets_tr.glob("*.json"))


def _pick_stratification_target(schema) -> str | None:
    for name, spec in schema.items():
        if spec.type is TargetType.CLASSIFICATION and spec.scope.value == "per_case":
            return name
    for name, spec in schema.items():
        if spec.type is TargetType.CLASSIFICATION:
            return name
    for name, spec in schema.items():
        if spec.type is TargetType.REGRESSION:
            return name
    return None


def _strat_labels(
    basenames: list[str],
    all_targets: dict[str, Any],
    schema,
    stratification_target: str | None,
) -> np.ndarray | None:
    if not stratification_target or stratification_target not in schema:
        return None
    spec = schema[stratification_target]
    raw = np.asarray(
        [_strat_value(all_targets[name], stratification_target, schema) for name in basenames]
    )
    if spec.type is TargetType.REGRESSION:
        quantiles = np.quantile(raw, [0.25, 0.5, 0.75])
        y = np.digitize(raw, quantiles, right=True).astype(int)
    else:
        y = raw.astype(int)
    counts = np.bincount(y - y.min()) if y.size else np.array([])
    if counts.size == 0 or int(counts.min()) < 2:
        return None
    return y


def _split_holdout(
    basenames: list[str],
    y: np.ndarray | None,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], str]:
    names = np.asarray(sorted(basenames))
    n = len(names)
    if n < 2:
        raise ValueError("Need at least two cases in targetsTr to create a holdout split")
    holdout_n = int(round(n * holdout_fraction))
    holdout_n = max(1, min(n - 1, holdout_n))

    if y is not None:
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=holdout_n, random_state=seed
            )
            train_idx, holdout_idx = next(splitter.split(names, y))
            return (
                [str(v) for v in names[train_idx]],
                [str(v) for v in names[holdout_idx]],
                "stratified_shuffle",
            )
        except ValueError:
            pass

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    holdout_idx = perm[:holdout_n]
    train_idx = perm[holdout_n:]
    return (
        [str(v) for v in sorted(names[train_idx])],
        [str(v) for v in sorted(names[holdout_idx])],
        "random_shuffle",
    )


def _safe_move(src: Path, dst: Path) -> str:
    if not src.exists():
        return "missing_source"
    if dst.exists():
        return "skipped_exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return "moved"


def _move_case_files(paths: ScatterRadPaths, case_id: str) -> dict[str, dict[str, int]]:
    stats = {
        "images": {"moved": 0, "skipped_exists": 0, "missing_source": 0},
        "labels": {"moved": 0, "skipped_exists": 0, "missing_source": 0},
        "targets": {"moved": 0, "skipped_exists": 0, "missing_source": 0},
    }

    image_files = sorted(paths.images_tr.glob(f"{case_id}_*.nii.gz"))
    if not image_files:
        stats["images"]["missing_source"] += 1
    for src in image_files:
        outcome = _safe_move(src, paths.images_ts / src.name)
        stats["images"][outcome] += 1

    label_candidates = [
        *sorted(paths.labels_tr.glob(f"{case_id}.nii.gz")),
        *sorted(paths.labels_tr.glob(f"{case_id}_*.nii.gz")),
    ]
    if not label_candidates:
        stats["labels"]["missing_source"] += 1
    seen = set()
    for src in label_candidates:
        if src.name in seen:
            continue
        seen.add(src.name)
        outcome = _safe_move(src, paths.labels_ts / src.name)
        stats["labels"][outcome] += 1

    target_file = paths.targets_tr / f"{case_id}.json"
    outcome = _safe_move(target_file, paths.targets_ts / target_file.name)
    stats["targets"][outcome] += 1
    return stats


def generate_holdout(
    paths: ScatterRadPaths,
    holdout_fraction: float = 0.2,
    seed: int = 42,
    move_files: bool = True,
) -> dict[str, Any]:
    """Generate a deterministic holdout split from targetsTr and optionally move files to *Ts."""

    if holdout_fraction <= 0.0 or holdout_fraction >= 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")

    schema = load_targets_schema(paths.targets_json)
    basenames = _target_basenames(paths.targets_tr)
    if not basenames:
        raise ValueError(f"No target files found in {paths.targets_tr}")
    all_targets = {
        b: load_case_targets(paths.targets_tr / f"{b}.json", schema=schema, basename=b)
        for b in basenames
    }
    stratification_target = _pick_stratification_target(schema)
    y = _strat_labels(basenames, all_targets, schema, stratification_target)
    train_cases, holdout_cases, split_strategy = _split_holdout(
        basenames=basenames,
        y=y,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )

    move_summary = {
        "images": {"moved": 0, "skipped_exists": 0, "missing_source": 0},
        "labels": {"moved": 0, "skipped_exists": 0, "missing_source": 0},
        "targets": {"moved": 0, "skipped_exists": 0, "missing_source": 0},
    }
    if move_files:
        for case_id in holdout_cases:
            stats = _move_case_files(paths, case_id)
            for group, group_stats in stats.items():
                for key, value in group_stats.items():
                    move_summary[group][key] += int(value)

    payload = {
        "seed": seed,
        "source": "targetsTr",
        "holdout_fraction": holdout_fraction,
        "strategy": split_strategy,
        "stratification_key": stratification_target,
        "n_cases": len(basenames),
        "train_cases": sorted(train_cases),
        "holdout_cases": sorted(holdout_cases),
        "moved_to_ts": move_summary,
    }
    out_path = paths.raw_dataset_dir / "test_split_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    log.info("Wrote holdout manifest to %s", out_path)
    return payload
