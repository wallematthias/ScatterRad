from __future__ import annotations

import pandas as pd
from pathlib import Path
import re
import joblib

from scatterrad.paths import ScatterRadPaths


def predict(
    paths: ScatterRadPaths,
    task_name: str,
    model_kind: str,
    inputs: list[Path],
    fold: int | None = None,
) -> pd.DataFrame:
    """Run radiomics inference using saved model(s)."""

    if model_kind != "radiomics":
        raise ValueError("radiomics predictor called with non-radiomics model")
    if not inputs:
        return pd.DataFrame(columns=["input", "prediction"])

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
    preds = []
    for input_path in inputs:
        features = pd.DataFrame([{}])
        fold_preds = []
        for fold_idx in folds:
            payload = joblib.load(paths.result_dir(task_name, model_kind, fold_idx) / "model.joblib")
            model = payload.get("model", payload) if isinstance(payload, dict) else payload
            fold_preds.append(float(model.predict(features)[0]))
        preds.append(
            {"input": str(input_path), "prediction": float(sum(fold_preds) / len(fold_preds))}
        )
    return pd.DataFrame(preds)
