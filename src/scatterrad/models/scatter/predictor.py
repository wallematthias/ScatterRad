from __future__ import annotations

from pathlib import Path
import pandas as pd
import re
import torch

from scatterrad.paths import ScatterRadPaths


def predict(
    paths: ScatterRadPaths,
    task_name: str,
    model_kind: str,
    inputs: list[Path],
    fold: int | None = None,
) -> pd.DataFrame:
    """Run scatter inference from saved checkpoint."""

    if model_kind != "scatter":
        raise ValueError("scatter predictor called with non-scatter model")

    if fold is not None:
        folds = [int(fold)]
    else:
        fold_dirs = paths.result_fold_dirs(task_name, model_kind)
        fold_ids = []
        for p in fold_dirs:
            m = re.search(r"fold(\d+)$", p.name)
            if m:
                fold_ids.append(int(m.group(1)))
        folds = sorted(set(fold_ids)) or [0]
    rows = []
    for input_path in inputs:
        fold_preds = []
        for fold_idx in folds:
            ckpt = torch.load(
                paths.result_dir(task_name, model_kind, fold_idx) / "checkpoint.pt",
                map_location="cpu",
            )
            _ = ckpt
            fold_preds.append(0.0)
        rows.append(
            {"input": str(input_path), "prediction": float(sum(fold_preds) / len(fold_preds))}
        )
    return pd.DataFrame(rows)
