from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics

from scatterrad.config import TargetType


def _f(value: float) -> float:
    return float(value) if not isinstance(value, float) or not math.isnan(value) else math.nan


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    target_type: TargetType,
    num_classes: int | None = None,
) -> dict[str, float | list]:
    """Compute task metrics for classification or regression."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if target_type is TargetType.REGRESSION:
        mae = metrics.mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
        r2 = metrics.r2_score(y_true, y_pred)
        try:
            pear = pearsonr(y_true, y_pred).statistic
        except Exception:
            pear = math.nan
        try:
            spear = spearmanr(y_true, y_pred).statistic
        except Exception:
            spear = math.nan
        return {
            "mae": _f(mae),
            "rmse": _f(rmse),
            "r2": _f(r2),
            "pearson": _f(float(pear) if pear is not None else math.nan),
            "spearman": _f(float(spear) if spear is not None else math.nan),
        }

    out: dict[str, float | list] = {
        "balanced_accuracy": _f(metrics.balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": metrics.confusion_matrix(y_true, y_pred).tolist(),
    }
    classes = int(num_classes or 2)
    if classes == 2:
        out["f1"] = _f(metrics.f1_score(y_true, y_pred, zero_division=0))
        out["precision"] = _f(metrics.precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = _f(metrics.recall_score(y_true, y_pred, zero_division=0))
        if y_proba is not None and len(np.unique(y_true)) > 1:
            out["auc"] = _f(metrics.roc_auc_score(y_true, y_proba[:, 1]))
            out["brier_score"] = _f(metrics.brier_score_loss(y_true, y_proba[:, 1]))
        else:
            out["auc"] = math.nan
            out["brier_score"] = math.nan
        return out

    out["f1_macro"] = _f(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1_weighted"] = _f(metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["per_class_f1"] = [
        _f(v) for v in metrics.f1_score(y_true, y_pred, average=None, zero_division=0)
    ]
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["auc_macro"] = _f(
            metrics.roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
    else:
        out["auc_macro"] = math.nan
    return out
