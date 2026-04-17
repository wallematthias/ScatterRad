from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from scatterrad.config import (
    DatasetConfig,
    ModelKind,
    PlansConfig,
    TaskConfig,
    TargetScope,
    TargetType,
    TargetsSchema,
    load_case_targets,
)
from scatterrad.evaluation import compute_metrics
from scatterrad.models.radiomics.extractor import extract_all
from scatterrad.paths import ScatterRadPaths

log = logging.getLogger(__name__)



def _build_test_feature_matrix(
    paths: ScatterRadPaths,
    task: TaskConfig,
    schema: TargetsSchema,
    labels: tuple[int, ...],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build feature matrix from the test-set radiomics dir."""
    spec = schema[task.target]

    ts_targets_dir = paths.preprocessed_dataset_dir / "targets_ts"
    all_targets: dict[str, Any] = {}
    for target_file in sorted(ts_targets_dir.glob("*.json")):
        basename = target_file.stem
        all_targets[basename] = load_case_targets(target_file, schema=schema, basename=basename)

    if not all_targets:
        raise ValueError(f"No test-set target files found in {ts_targets_dir}")

    rad_dir = paths.radiomics_ts_dir

    if spec.scope is TargetScope.PER_LABEL:
        rows, y, ids = [], [], []
        for feature_file in sorted(rad_dir.glob("*.json")):
            stem = feature_file.stem
            if "_label" not in stem:
                continue
            basename, label_part = stem.split("_label")
            label_id = int(label_part)
            if label_id not in labels:
                continue
            payload = json.loads(feature_file.read_text())
            features = payload.get("features", {})
            if basename not in all_targets:
                continue
            target = all_targets[basename].get_per_label(task.target, label_id)
            if math.isnan(target):
                continue
            rows.append(features)
            y.append(int(target) if spec.type is TargetType.CLASSIFICATION else float(target))
            ids.append(f"{basename}__label{label_id}")
        return pd.DataFrame(rows), np.asarray(y), ids

    case_rows: dict[str, dict[str, float]] = {}
    for feature_file in sorted(rad_dir.glob("*.json")):
        stem = feature_file.stem
        if "_label" not in stem:
            continue
        basename, label_part = stem.split("_label")
        label_id = int(label_part)
        if label_id not in labels:
            continue
        payload = json.loads(feature_file.read_text())
        features = payload.get("features", {})
        row = case_rows.setdefault(basename, {})
        for key, val in features.items():
            row[f"label{label_id:03d}_{key}"] = float(val)

    rows, y, ids = [], [], []
    for basename, row in sorted(case_rows.items()):
        if basename not in all_targets:
            continue
        target = all_targets[basename].get_per_case(task.target)
        if math.isnan(target):
            continue
        rows.append(row)
        y.append(int(target) if spec.type is TargetType.CLASSIFICATION else float(target))
        ids.append(basename)
    return pd.DataFrame(rows), np.asarray(y), ids


def test(
    paths: ScatterRadPaths,
    task: TaskConfig,
    dataset: DatasetConfig,
    schema: TargetsSchema,
    plans: PlansConfig,
) -> None:
    """Run ensemble test-set evaluation using all trained fold models."""

    if task.model is not ModelKind.RADIOMICS:
        raise ValueError("Radiomics tester called for non-radiomics task")

    log.info("Task: %s  target: %s  model: radiomics (test)", task.name, task.target)

    # Preprocess test set if crops are missing
    crops_ts_dir = paths.preprocessed_dataset_dir / "crops_ts"
    targets_ts_dir = paths.preprocessed_dataset_dir / "targets_ts"
    if not crops_ts_dir.exists() or not any(crops_ts_dir.glob("*.npz")):
        log.info("Test crops not found — running test-set preprocessing ...")
        from scatterrad.preprocessing.runner import preprocess_test
        import os as _os
        _workers = int(_os.environ.get("SCATTERRAD_NP", -1))
        preprocess_test(paths, num_workers=_workers)
    else:
        log.info("Test crops found at %s", crops_ts_dir)

    # Extract test-set radiomics into radiomics_ts_dir
    log.info("Extracting test-set radiomics features ...")
    import os as _os
    _workers = int(_os.environ.get("SCATTERRAD_NP", -1))
    extract_all(
        paths,
        modality=dataset.modality,
        num_workers=_workers,
        force=False,
        crops_dir=paths.crops_ts_dir,
        output_dir=paths.radiomics_ts_dir,
    )

    labels = task.resolved_labels(schema)
    log.info("Building test feature matrix for labels %s ...", sorted(labels))
    x_test_raw, y_test, ids_test = _build_test_feature_matrix(paths, task, schema, labels=labels)
    if x_test_raw.empty:
        raise ValueError("No test samples available")
    log.info("Test feature matrix: %d samples x %d features", x_test_raw.shape[0], x_test_raw.shape[1])

    # Discover fold result dirs
    fold_dirs = paths.result_fold_dirs(task.name, "radiomics")
    if not fold_dirs:
        raise ValueError(
            f"No trained fold models found for task '{task.name}'. Run 'scatterrad train' first."
        )
    log.info("Found %d fold model(s): %s", len(fold_dirs), [p.name for p in fold_dirs])

    spec = schema[task.target]

    # Ensemble: average predictions / probabilities across folds
    all_preds: list[np.ndarray] = []
    all_probas: list[np.ndarray] = []

    for fold_dir in fold_dirs:
        model_path = fold_dir / "model.joblib"
        if not model_path.exists():
            log.warning("Skipping %s: model.joblib not found", fold_dir)
            continue

        bundle = joblib.load(model_path)
        imputer = bundle["imputer"]
        scaler = bundle["scaler"]
        selected_features = bundle["selected_feature_names"]
        model = bundle["model"]

        # Reconstruct the pipeline in the same order as training:
        #   1. align to imputer input features (pre-selection)
        #   2. impute + scale
        #   3. select down to the features the model expects
        imputer_features = bundle.get("imputer_feature_names") or list(
            imputer.feature_names_in_ if hasattr(imputer, "feature_names_in_") else selected_features
        )
        x = x_test_raw.reindex(columns=imputer_features)
        x_imp = imputer.transform(x)
        x_s = scaler.transform(x_imp)
        # Select the columns the model was trained on
        imputer_feat_index = {f: i for i, f in enumerate(imputer_features)}
        sel_idx = [imputer_feat_index[f] for f in selected_features if f in imputer_feat_index]
        x_s = x_s[:, sel_idx]

        fold_pred = model.predict(x_s)
        all_preds.append(fold_pred)

        if spec.type is TargetType.CLASSIFICATION and hasattr(model, "predict_proba"):
            all_probas.append(model.predict_proba(x_s))

    if not all_preds:
        raise ValueError("No valid fold models produced predictions")

    # Ensemble by averaging
    y_pred_ensemble = np.mean(np.stack(all_preds, axis=0), axis=0)
    if spec.type is TargetType.CLASSIFICATION:
        y_pred_hard = np.round(y_pred_ensemble).astype(int)
    else:
        y_pred_hard = y_pred_ensemble

    y_proba_ensemble: np.ndarray | None = None
    if all_probas:
        y_proba_ensemble = np.mean(np.stack(all_probas, axis=0), axis=0)

    metrics = compute_metrics(
        y_test, y_pred_hard, y_proba_ensemble, target_type=spec.type, num_classes=spec.num_classes
    )

    metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
    log.info("Test set results (%d samples, %d-fold ensemble): %s", len(y_test), len(all_preds), metrics_str)

    # Write results
    result_dir = paths.result_config_dir(
        task.name,
        model_kind="radiomics",
        planner_name=plans.planner,
        trainer_name="scatterradDefaultTrainer",
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({"sample_id": ids_test, "y_true": y_test, "y_pred": y_pred_hard})
    if y_proba_ensemble is not None:
        for class_idx in range(y_proba_ensemble.shape[1]):
            pred_df[f"p{class_idx}"] = y_proba_ensemble[:, class_idx]
    pred_df.to_csv(result_dir / "test_predictions.csv", index=False)

    payload = {
        "task": task.name,
        "model": "radiomics",
        "split": "test",
        "n_folds_ensemble": len(all_preds),
        "n_test": int(len(y_test)),
        "target_type": spec.type.value,
        "metrics": metrics,
    }
    (result_dir / "test_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    log.info("Results written to %s", result_dir)
