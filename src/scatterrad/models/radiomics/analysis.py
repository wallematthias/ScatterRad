from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scatterrad.paths import ScatterRadPaths


def _load_feature_table(paths: ScatterRadPaths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_file in sorted(paths.radiomics_dir.glob("*.json")):
        payload = json.loads(feature_file.read_text())
        features = payload.get("features", {})
        if not isinstance(features, dict):
            continue
        row = {"sample_id": feature_file.stem}
        for key, val in features.items():
            try:
                row[str(key)] = float(val)
            except (TypeError, ValueError):
                row[str(key)] = float("nan")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["sample_id"])
    return pd.DataFrame(rows)


def compute_intercorrelation(
    paths: ScatterRadPaths,
    corr_threshold: float = 0.9,
) -> dict[str, Any]:
    """Compute dataset-level feature intercorrelation reports from radiomics cache."""

    df = _load_feature_table(paths)
    out_dir = paths.preprocessed_dataset_dir / "radiomics_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty or len(df.columns) <= 1:
        summary = {
            "n_samples": int(len(df)),
            "n_features": 0,
            "corr_threshold": float(corr_threshold),
            "n_high_corr_pairs": 0,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    x = df.drop(columns=["sample_id"]).replace([np.inf, -np.inf], np.nan)
    keep = [c for c in x.columns if x[c].notna().any() and x[c].nunique(dropna=True) > 1]
    x = x.loc[:, keep]
    if x.empty:
        summary = {
            "n_samples": int(len(df)),
            "n_features": 0,
            "corr_threshold": float(corr_threshold),
            "n_high_corr_pairs": 0,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    med = x.median(axis=0, numeric_only=True)
    x = x.fillna(med)
    corr = x.corr(method="spearman")

    corr.to_csv(out_dir / "spearman_corr.csv")

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = []
    for col in upper.columns:
        vals = upper[col]
        high = vals[vals.abs() > float(corr_threshold)]
        for row_name, value in high.items():
            pairs.append((str(row_name), str(col), float(value)))
    pairs_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman_r"])
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(by="spearman_r", key=lambda s: s.abs(), ascending=False)
    pairs_df.to_csv(out_dir / "high_corr_pairs.csv", index=False)

    summary = {
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "corr_threshold": float(corr_threshold),
        "n_high_corr_pairs": int(len(pairs_df)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
