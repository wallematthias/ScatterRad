from __future__ import annotations

import math

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from scatterrad.config import CaseTargets, TargetScope, TargetType, TargetsSchema


def _strat_value(case: CaseTargets, name: str, schema: TargetsSchema) -> int | float:
    spec = schema[name]
    if spec.scope is TargetScope.PER_CASE:
        val = case.get_per_case(name)
        return val if not math.isnan(val) else -1
    values = [case.get_per_label(name, lid) for lid in spec.applicable_labels]
    valid = [v for v in values if not math.isnan(v)]
    if spec.type is TargetType.CLASSIFICATION and spec.num_classes == 2:
        return int(any(v > 0.5 for v in valid))
    if not valid:
        return -1
    return float(np.nanmean(valid))


def generate_splits(
    basenames: list[str],
    targets: dict[str, CaseTargets],
    schema: TargetsSchema,
    n_folds: int = 5,
    seed: int = 42,
    stratification_target: str | None = None,
) -> list[dict[str, list[str]]]:
    """Deterministic patient-level folds with optional stratification."""

    if n_folds < 2:
        raise ValueError("n_folds must be >=2")
    if len(basenames) < n_folds:
        raise ValueError("n_folds cannot exceed number of cases")

    names = np.asarray(sorted(basenames))
    y = None
    use_stratified = False
    if stratification_target and stratification_target in schema:
        raw = np.asarray(
            [_strat_value(targets[name], stratification_target, schema) for name in names]
        )
        spec = schema[stratification_target]
        if spec.type is TargetType.REGRESSION:
            quantiles = np.quantile(raw, [0.25, 0.5, 0.75])
            y = np.digitize(raw, quantiles, right=True)
        else:
            y = raw.astype(int)
        use_stratified = len(np.unique(y)) > 1

    if use_stratified:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        indices = splitter.split(names, y)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        indices = splitter.split(names)

    folds: list[dict[str, list[str]]] = []
    for train_idx, val_idx in indices:
        folds.append(
            {
                "train": [str(x) for x in names[train_idx]],
                "val": [str(x) for x in names[val_idx]],
            }
        )
    return folds
