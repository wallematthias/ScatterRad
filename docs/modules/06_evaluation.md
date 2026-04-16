# Module 06: Evaluation

**Status:** Not implemented. Phase 6.

**Depends on:** `config/` only.

**Used by:** `models/radiomics/trainer.py`, `models/scatter/trainer.py`, CLI `report`.

---

## Overview

Metrics computation (fold-level) and aggregation (task-level). Shared by
both tracks. No ML logic — pure scoring.

---

## File: `src/scatterrad/evaluation/metrics.py`

### Public API

```python
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,          # class labels for classification, values for regression
    y_proba: np.ndarray | None,  # [N, num_classes] for classification, None for regression
    target_type: TargetType,
    num_classes: int | None = None,
) -> dict[str, float | list]:
    """Returns a dict suitable for metrics.json."""
```

### Classification metrics

Binary (num_classes==2):
- `auc` — ROC AUC (sklearn.metrics.roc_auc_score)
- `balanced_accuracy`
- `f1`
- `precision`, `recall`
- `confusion_matrix` — 2x2 list
- `brier_score` — calibration

Multiclass (num_classes>2):
- `auc_macro` — one-vs-rest, macro-averaged
- `balanced_accuracy`
- `f1_macro`, `f1_weighted`
- `confusion_matrix` — nxn list
- `per_class_f1` — list of floats

Handle edge cases:
- Only one class in `y_true` → return NaN for AUC, document in log.
- y_proba is None → skip AUC-based metrics, set to NaN.

### Regression metrics

- `mae` — mean absolute error
- `rmse` — root mean squared error
- `r2` — coefficient of determination
- `pearson` — Pearson correlation (scipy.stats.pearsonr)
- `spearman` — Spearman correlation

### Implementation

All using `sklearn.metrics` where possible. No custom loops.

### Tests

- Perfect predictions → auc=1, mae=0, etc.
- Random predictions → auc ≈ 0.5, r2 ≈ 0.
- Single-class edge case → NaN for AUC, not a crash.
- All metric values are JSON-serializable (no numpy types leaking).

---

## File: `src/scatterrad/evaluation/aggregate.py`

### Public API

```python
def aggregate_folds(
    result_dirs: list[Path],  # one per fold
) -> dict:
    """Read each fold's metrics.json, compute mean ± std across folds."""
```

Returns:
```python
{
    "n_folds": 5,
    "metrics_per_fold": [ {...fold 0 metrics...}, ... ],
    "metrics_summary": {
        "auc": {"mean": 0.82, "std": 0.04, "min": 0.77, "max": 0.88},
        ...
    },
    "attention_weights_mean": {...},  # if scatter + per_case
    "total_runtime_seconds": 1823.0,
}
```

### Tests

- Summary stats correct on fixture data.
- Robust to a fold with NaN metrics (e.g., single-class val fold).

---

## File: `src/scatterrad/evaluation/report.py`

### Public API

```python
def render_report(
    paths: ScatterRadPaths,
    task_name: str,
    model_kind: str,
    output_path: Path | None = None,  # default: stdout
) -> str:
    """Render a markdown report. Also writes to results/<task>__<model>/report.md."""
```

Contents:
- Task summary (target, model, num folds, num train/val per fold).
- Metrics summary table (one row per metric, columns = mean, std, range).
- Per-fold table.
- For scatter + per_case: mean attention weight per label, with bar
  visualization (markdown).
- Run timing.

Keep it readable in the terminal (plain markdown). No embedded images in v1.

---

## File: `src/scatterrad/evaluation/plots.py` (optional, v1.1)

If time permits:
- ROC curve per fold (matplotlib).
- Bland-Altman plot for regression.
- Calibration curve.

All saved as PNG alongside `report.md`. Deferred to Phase 6+ if complex.

---

## Invariants

1. Metrics are computed identically for both tracks (same function, same
   definitions).
2. All floats written to JSON are Python floats, not numpy scalars.
3. NaN metrics are allowed (documented, not crashed on).

---

## What NOT to do

- Do not introduce per-model-kind metrics. Both tracks report the same set.
- Do not compute metrics over the entire dataset pretending it was one fold —
  always per-fold, then aggregate. (Leakage risk + misleading summaries.)
