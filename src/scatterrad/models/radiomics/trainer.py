from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

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
from scatterrad.utils import resolve_num_workers

log = logging.getLogger(__name__)


def _load_splits(path: Path) -> list[dict[str, list[str]]]:
    payload = json.loads(path.read_text())
    return payload["folds"]


def _load_case_targets(paths: ScatterRadPaths, schema: TargetsSchema) -> dict[str, Any]:
    out = {}
    for target_file in sorted(paths.training_targets_tr.glob("*.json")):
        basename = target_file.stem
        out[basename] = load_case_targets(target_file, schema=schema, basename=basename)
    return out


def build_feature_matrix(
    paths: ScatterRadPaths,
    task: TaskConfig,
    schema: TargetsSchema,
    labels: tuple[int, ...],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build feature matrix and targets for one task."""

    spec = schema[task.target]
    all_targets = _load_case_targets(paths, schema)

    if spec.scope is TargetScope.PER_LABEL:
        rows = []
        y = []
        ids = []
        for feature_file in sorted(paths.radiomics_dir.glob("*.json")):
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
    for feature_file in sorted(paths.radiomics_dir.glob("*.json")):
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

    rows = []
    y = []
    ids = []
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


def _sanitize_feature_matrix(x: pd.DataFrame) -> pd.DataFrame:
    x2 = x.copy()
    x2 = x2.replace([np.inf, -np.inf], np.nan)
    keep = [c for c in x2.columns if x2[c].notna().any()]
    return x2.loc[:, keep]


def _load_icc_scores(path: Path) -> dict[str, float]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        return {str(k): float(v) for k, v in payload.items()}
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "feature" not in df.columns or "icc" not in df.columns:
            raise ValueError("ICC csv must have columns: feature, icc")
        return {str(k): float(v) for k, v in zip(df["feature"], df["icc"]) if pd.notna(v)}
    raise ValueError(f"Unsupported ICC file format: {path}")


def _global_pre_cv_feature_filter(
    x: pd.DataFrame,
    task: TaskConfig,
    paths: ScatterRadPaths,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pre-CV filtering with no label usage: invalid/constant/very-low-variance + optional ICC."""

    details: dict[str, Any] = {}
    x2 = _sanitize_feature_matrix(x)
    details["n_features_initial"] = int(x.shape[1])
    details["n_features_after_invalid"] = int(x2.shape[1])

    # Remove constant columns (after NaN-safe nunique computation)
    nun = x2.nunique(dropna=True)
    keep_const = nun[nun > 1].index.tolist()
    x2 = x2.loc[:, keep_const]
    details["n_features_after_constant"] = int(x2.shape[1])

    # Very low variance filtering (no labels), computed after median fill.
    global_var_thr = float(task.model_config.get("global_variance_threshold", 1e-8))
    if x2.shape[1] > 0:
        med = x2.median(axis=0, numeric_only=True)
        x_filled = x2.fillna(med)
        vt = VarianceThreshold(threshold=global_var_thr)
        vt.fit(x_filled.values)
        keep_mask = vt.get_support()
        keep_cols = [c for c, keep in zip(x2.columns, keep_mask) if bool(keep)]
        x2 = x2.loc[:, keep_cols]
    details["global_variance_threshold"] = global_var_thr
    details["n_features_after_global_variance"] = int(x2.shape[1])

    icc_path_raw = task.model_config.get("reproducibility_icc_path")
    if not icc_path_raw:
        default_icc = paths.preprocessed_dataset_dir / "radiomics_reproducibility" / "icc_scores.json"
        if default_icc.exists():
            icc_path_raw = str(default_icc)
    if icc_path_raw:
        icc_path = Path(str(icc_path_raw)).expanduser()
        if not icc_path.is_absolute():
            icc_path = Path.cwd() / icc_path
        if not icc_path.exists():
            raise FileNotFoundError(f"reproducibility_icc_path not found: {icc_path}")
        icc_threshold = float(task.model_config.get("reproducibility_icc_threshold", 0.75))
        icc_scores = _load_icc_scores(icc_path)

        _label_prefix = re.compile(r"^label\d+_")

        def _icc_lookup(col: str) -> float:
            score = icc_scores.get(col)
            if score is None:
                # Column may be prefixed with labelNNN_ (per-case aggregation)
                bare = _label_prefix.sub("", col, count=1)
                score = icc_scores.get(bare, -1.0)
            return float(score)

        keep_cols = [c for c in x2.columns if _icc_lookup(c) >= icc_threshold]
        x2 = x2.loc[:, keep_cols]
        details["reproducibility_icc_path"] = str(icc_path)
        details["reproducibility_icc_threshold"] = icc_threshold
    details["n_features_after_reproducibility"] = int(x2.shape[1])

    if x2.shape[1] == 0:
        log.error(
            "Pre-CV feature filter eliminated all features. Counts by stage: "
            "initial=%d, after_invalid=%d, after_constant=%d, after_global_variance=%d, "
            "after_reproducibility=%d  (global_variance_threshold=%s, icc_path=%s, icc_threshold=%s)",
            details["n_features_initial"],
            details["n_features_after_invalid"],
            details["n_features_after_constant"],
            details["n_features_after_global_variance"],
            details["n_features_after_reproducibility"],
            details["global_variance_threshold"],
            details.get("reproducibility_icc_path", "none"),
            details.get("reproducibility_icc_threshold", "n/a"),
        )

    return x2, details


def _fit_imputer_scaler(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)

    train_imp = imputer.fit_transform(x_train)
    val_imp = imputer.transform(x_val)

    train_s = scaler.fit_transform(train_imp)
    val_s = scaler.transform(val_imp)

    cols = list(x_train.columns)
    x_train_s = pd.DataFrame(train_s, columns=cols, index=x_train.index)
    x_val_s = pd.DataFrame(val_s, columns=cols, index=x_val.index)
    return x_train_s, x_val_s, imputer, scaler


def _variance_filter_train_only(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if x_train.shape[1] == 0:
        return x_train, x_val, []
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(x_train.values)
    keep_mask = vt.get_support()
    keep_cols = [c for c, keep in zip(x_train.columns, keep_mask) if bool(keep)]
    return x_train.loc[:, keep_cols], x_val.loc[:, keep_cols], keep_cols


def _correlation_prune_spearman(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if x_train.shape[1] <= 1:
        cols = list(x_train.columns)
        return x_train, x_val, cols

    corr = x_train.corr(method="spearman").abs().fillna(0.0)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    keep_cols = [c for c in x_train.columns if c not in to_drop]
    return x_train.loc[:, keep_cols], x_val.loc[:, keep_cols], keep_cols


def _feature_select(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    target_type: TargetType,
    seed: int,
    method: str,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if x_train.shape[1] == 0:
        return x_train, x_val, []
    method_norm = method.strip().lower()
    if method_norm in {"none", "off", "false"}:
        cols = list(x_train.columns)
        return x_train, x_val, cols

    k_eff = max(1, min(int(k), x_train.shape[1]))

    if method_norm == "univariate":
        score_fn = f_classif if target_type is TargetType.CLASSIFICATION else f_regression
        selector = SelectKBest(score_func=score_fn, k=k_eff)
        train_sel = selector.fit_transform(x_train.values, y_train)
        val_sel = selector.transform(x_val.values)
        keep = selector.get_support()
        cols = [c for c, keep_i in zip(x_train.columns, keep) if bool(keep_i)]
        return (
            pd.DataFrame(train_sel, columns=cols, index=x_train.index),
            pd.DataFrame(val_sel, columns=cols, index=x_val.index),
            cols,
        )

    if method_norm == "rf_importance":
        if target_type is TargetType.CLASSIFICATION:
            model = RandomForestClassifier(
                n_estimators=500,
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=int(os.environ.get("SCATTERRAD_NP", -1)),
            )
        else:
            model = RandomForestRegressor(
                n_estimators=500,
                random_state=seed,
                n_jobs=int(os.environ.get("SCATTERRAD_NP", -1)),
            )
        model.fit(x_train.values, y_train)
        rank = np.argsort(model.feature_importances_)[::-1][:k_eff]
        cols = [x_train.columns[i] for i in rank]
        return x_train.loc[:, cols], x_val.loc[:, cols], cols

    if method_norm == "rfe":
        if target_type is TargetType.CLASSIFICATION:
            estimator = RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=int(os.environ.get("SCATTERRAD_NP", -1)),
            )
        else:
            estimator = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=int(os.environ.get("SCATTERRAD_NP", -1)))
        selector = RFE(estimator=estimator, n_features_to_select=k_eff, step=0.1)
        train_sel = selector.fit_transform(x_train.values, y_train)
        val_sel = selector.transform(x_val.values)
        keep = selector.get_support()
        cols = [c for c, keep_i in zip(x_train.columns, keep) if bool(keep_i)]
        return (
            pd.DataFrame(train_sel, columns=cols, index=x_train.index),
            pd.DataFrame(val_sel, columns=cols, index=x_val.index),
            cols,
        )

    raise ValueError(f"Unsupported feature_selection method: {method}")


def _random_oversample(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    y = np.asarray(y_train)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) <= 1:
        return x_train, y_train
    max_count = int(counts.max())
    rng = np.random.default_rng(seed)

    idx_all = np.arange(len(y))
    res_idx = []
    for cls in classes:
        cls_idx = idx_all[y == cls]
        if len(cls_idx) == 0:
            continue
        if len(cls_idx) < max_count:
            extra = rng.choice(cls_idx, size=max_count - len(cls_idx), replace=True)
            cls_idx = np.concatenate([cls_idx, extra])
        res_idx.append(cls_idx)
    full_idx = np.concatenate(res_idx)
    rng.shuffle(full_idx)
    return x_train.iloc[full_idx].reset_index(drop=True), y[full_idx]


def _rf_estimator(target_type: TargetType, seed: int, class_weight: str | None):
    if target_type is TargetType.CLASSIFICATION:
        return RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight=class_weight,
            n_jobs=int(os.environ.get("SCATTERRAD_NP", -1)),
        )
    return RandomForestRegressor(
        n_estimators=500,
        random_state=seed,
        n_jobs=int(os.environ.get("SCATTERRAD_NP", -1)),
    )


def _fit_rf_with_tuning(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    target_type: TargetType,
    seed: int,
    class_weight: str | None,
    tune: bool,
    n_iter: int,
    inner_folds: int,
):
    base = _rf_estimator(target_type, seed, class_weight)
    if not tune:
        base.fit(x_train.values, y_train)
        return base, {"tuned": False}

    n = len(y_train)
    if n < 8:
        base.fit(x_train.values, y_train)
        return base, {"tuned": False, "reason": "too_few_samples"}

    if target_type is TargetType.CLASSIFICATION:
        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) <= 1 or int(counts.min()) < 2:
            base.fit(x_train.values, y_train)
            return base, {"tuned": False, "reason": "insufficient_class_diversity"}
        cv_splits = max(2, min(int(inner_folds), int(counts.min())))
        splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        if len(unique) == 2:
            scoring = "roc_auc"
        else:
            scoring = "roc_auc_ovr_weighted"
    else:
        cv_splits = max(2, min(int(inner_folds), int(n // 4)))
        splitter = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        scoring = "neg_mean_absolute_error"

    # Use fewer trees during search; the final refit uses the base estimator's n_estimators.
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 8, 12, 16, 24],
        "min_samples_split": [2, 4, 8, 16],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=max(1, int(n_iter)),
        scoring=scoring,
        cv=splitter,
        random_state=seed,
        n_jobs=int(os.environ.get("SCATTERRAD_NP", -1)),
        refit=True,
    )
    search.fit(x_train.values, y_train)
    return search.best_estimator_, {
        "tuned": True,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "inner_cv_splits": int(cv_splits),
    }


def _write_aggregated_importances(
    paths: ScatterRadPaths,
    task: TaskConfig,
    plans: PlansConfig,
) -> None:
    """Average feature importances across folds and write a summary CSV."""
    fold_dirs = paths.result_fold_dirs(task.name, "radiomics")
    accum: dict[str, list[float]] = {}
    for fold_dir in fold_dirs:
        imp_path = fold_dir / "feature_importances.csv"
        if not imp_path.exists():
            continue
        df = pd.read_csv(imp_path)
        for _, row in df.iterrows():
            accum.setdefault(str(row["feature"]), []).append(float(row["importance"]))

    if not accum:
        return

    rows = [
        {"feature": feat, "importance_mean": float(np.mean(vals)), "importance_std": float(np.std(vals)), "n_folds": len(vals)}
        for feat, vals in accum.items()
    ]
    agg_df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    out_dir = paths.result_config_dir(
        task.name,
        model_kind="radiomics",
        planner_name=plans.planner,
        trainer_name="scatterradDefaultTrainer",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_importances.csv"
    agg_df.to_csv(out_path, index=False)
    log.info(
        "Feature importances aggregated across %d fold(s) → %s",
        len(fold_dirs), out_path,
    )


def train(
    paths: ScatterRadPaths,
    task: TaskConfig,
    dataset: DatasetConfig,
    schema: TargetsSchema,
    plans: PlansConfig,
    fold: int | None = None,
    continue_existing: bool = False,
) -> None:
    """Train radiomics models on folds and write results artifacts."""

    if task.model is not ModelKind.RADIOMICS:
        raise ValueError("Radiomics trainer called for non-radiomics task")

    log.info("Task: %s  target: %s  model: radiomics", task.name, task.target)

    rx_workers = resolve_num_workers(task.model_config.get("num_workers", 0))
    log.info("Extracting radiomics features (num_workers=%d) ...", rx_workers)
    extract_all(paths, modality=dataset.modality, num_workers=rx_workers, force=False)

    labels = task.resolved_labels(schema)
    log.info("Building feature matrix for labels %s ...", sorted(labels))
    x_all_raw, y_all, ids_all = build_feature_matrix(paths, task, schema, labels=labels)
    if x_all_raw.empty:
        raise ValueError("No training samples available after filtering")
    log.info("Feature matrix: %d samples x %d features", x_all_raw.shape[0], x_all_raw.shape[1])

    log.info("Running pre-CV feature filter ...")
    x_all, global_filter_details = _global_pre_cv_feature_filter(x_all_raw, task, paths=paths)
    if x_all.shape[1] == 0:
        raise ValueError("No usable radiomics features remain after pre-CV filtering")
    log.info(
        "Pre-CV filter: %d → %d features  (removed %d)",
        x_all_raw.shape[1],
        x_all.shape[1],
        x_all_raw.shape[1] - x_all.shape[1],
    )

    splits_path = paths.results_splits_json if paths.results_splits_json.exists() else paths.splits_json
    folds = _load_splits(splits_path)
    selected = range(len(folds)) if fold is None else [int(fold)]
    spec = schema[task.target]
    log.info("Running %d fold(s) of %d total", len(list(selected)), len(folds))
    selected = range(len(folds)) if fold is None else [int(fold)]

    for fold_idx in selected:
        result_dir = paths.result_dir(
            task.name,
            "radiomics",
            fold_idx,
            planner_name=plans.planner,
            trainer_name="scatterradDefaultTrainer",
        )
        if continue_existing and (result_dir / "metrics.json").exists():
            log.info("Fold %d/%d: skipping (already done)", fold_idx, len(folds) - 1)
            continue

        log.info("Fold %d/%d: preparing split ...", fold_idx, len(folds) - 1)
        fold_info = folds[fold_idx]
        train_case = set(fold_info["train"])
        val_case = set(fold_info["val"])

        def _case_of(sample_id: str) -> str:
            return sample_id.split("__")[0]

        train_mask = np.asarray([_case_of(sid) in train_case for sid in ids_all])
        val_mask = np.asarray([_case_of(sid) in val_case for sid in ids_all])

        x_train_raw = x_all.loc[train_mask].copy()
        x_val_raw = x_all.loc[val_mask].copy()
        y_train = y_all[train_mask]
        y_val = y_all[val_mask]
        ids_val = [sid for sid, keep in zip(ids_all, val_mask) if keep]

        if x_train_raw.empty or x_val_raw.empty:
            log.warning("Skipping fold %d due to empty split", fold_idx)
            continue

        log.info(
            "Fold %d/%d: %d train samples, %d val samples",
            fold_idx, len(folds) - 1, len(y_train), len(y_val),
        )
        t0 = perf_counter()

        x_train, x_val, imputer, scaler = _fit_imputer_scaler(x_train_raw, x_val_raw)

        var_thr = float(task.model_config.get("variance_threshold", 1e-6))
        x_train, x_val, cols_after_var = _variance_filter_train_only(x_train, x_val, var_thr)
        log.info("Fold %d/%d: after variance filter: %d features", fold_idx, len(folds) - 1, len(cols_after_var))

        corr_thr = float(task.model_config.get("corr_threshold", 0.9))
        log.info("Fold %d/%d: running Spearman correlation pruning (threshold=%.2f) ...", fold_idx, len(folds) - 1, corr_thr)
        x_train, x_val, cols_after_corr = _correlation_prune_spearman(x_train, x_val, corr_thr)
        log.info("Fold %d/%d: after correlation pruning: %d features", fold_idx, len(folds) - 1, len(cols_after_corr))

        fs_method = str(task.model_config.get("feature_selection", "univariate"))
        fs_k = int(task.model_config.get("feature_selection_k", min(128, max(1, x_train.shape[1]))))
        log.info("Fold %d/%d: feature selection (method=%s, k=%d) ...", fold_idx, len(folds) - 1, fs_method, fs_k)
        x_train, x_val, cols_after_fs = _feature_select(
            x_train,
            y_train,
            x_val,
            target_type=spec.type,
            seed=task.cv.seed,
            method=fs_method,
            k=fs_k,
        )
        if x_train.shape[1] == 0:
            raise ValueError("No features left after fold-level filtering/selection")
        log.info("Fold %d/%d: final feature count: %d", fold_idx, len(folds) - 1, x_train.shape[1])

        imbalance = str(task.model_config.get("imbalance", "class_weight")).lower()
        class_weight: str | None = None
        x_train_fit = x_train
        y_train_fit = y_train
        if spec.type is TargetType.CLASSIFICATION:
            if imbalance in {"class_weight", "balanced", "balanced_subsample"}:
                class_weight = "balanced_subsample"
            elif imbalance in {"oversample", "random_oversample"}:
                x_train_fit, y_train_fit = _random_oversample(x_train, y_train, seed=task.cv.seed)
            elif imbalance in {"none", "off"}:
                class_weight = None
            else:
                raise ValueError(f"Unsupported imbalance strategy: {imbalance}")

        tune = bool(task.model_config.get("tune_hyperparameters", True))
        tune_iter = int(task.model_config.get("tune_n_iter", 20))
        inner_folds = int(task.model_config.get("tune_inner_folds", 3))

        log.info(
            "Fold %d/%d: fitting RF (tune=%s, n_iter=%d, inner_folds=%d) ...",
            fold_idx, len(folds) - 1, tune, tune_iter, inner_folds,
        )
        model, tune_info = _fit_rf_with_tuning(
            x_train_fit,
            y_train_fit,
            target_type=spec.type,
            seed=task.cv.seed,
            class_weight=class_weight,
            tune=tune,
            n_iter=tune_iter,
            inner_folds=inner_folds,
        )
        if tune_info.get("tuned"):
            log.info(
                "Fold %d/%d: best CV score=%.4f  params=%s",
                fold_idx, len(folds) - 1, tune_info["best_score"], tune_info["best_params"],
            )

        y_pred = model.predict(x_val.values)

        y_proba = None
        if spec.type is TargetType.CLASSIFICATION:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(x_val.values)

        metrics = compute_metrics(
            y_val, y_pred, y_proba, target_type=spec.type, num_classes=spec.num_classes
        )
        runtime = perf_counter() - t0

        result_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "imputer": imputer,
            "imputer_feature_names": list(x_train_raw.columns),
            "scaler": scaler,
            "selected_feature_names": list(x_train.columns),
            "model": model,
        }
        joblib.dump(bundle, result_dir / "model.joblib")

        pred_df = pd.DataFrame({"sample_id": ids_val, "y_true": y_val, "y_pred": y_pred})
        if y_proba is not None:
            for class_idx in range(y_proba.shape[1]):
                pred_df[f"p{class_idx}"] = y_proba[:, class_idx]
        pred_df.to_csv(result_dir / "predictions.csv", index=False)

        preprocessing_payload = {
            "global": global_filter_details,
            "fold": {
                "variance_threshold": var_thr,
                "corr_threshold": corr_thr,
                "feature_selection": fs_method,
                "feature_selection_k": fs_k,
                "n_features_after_variance": len(cols_after_var),
                "n_features_after_correlation": len(cols_after_corr),
                "n_features_after_selection": len(cols_after_fs),
                "imbalance": imbalance,
                "hyperparameter_tuning": tune_info,
            },
        }
        (result_dir / "preprocessing.json").write_text(
            json.dumps(preprocessing_payload, indent=2, sort_keys=True) + "\n"
        )

        payload = {
            "task": task.name,
            "model": "radiomics",
            "fold": fold_idx,
            "n_train": int(len(y_train_fit)),
            "n_val": int(len(y_val)),
            "target_type": spec.type.value,
            "metrics": metrics,
            "runtime_seconds": float(runtime),
        }
        (result_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        (result_dir / "features_used.txt").write_text("\n".join(map(str, x_train.columns)) + "\n")

        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        log.info(
            "Fold %d/%d done in %.1fs  →  %s",
            fold_idx, len(folds) - 1, runtime, metrics_str,
        )

        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({
                "feature": list(x_train.columns),
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(result_dir / "feature_importances.csv", index=False)

    _write_aggregated_importances(paths, task, plans)
