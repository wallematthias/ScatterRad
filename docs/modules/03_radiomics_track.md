# Module 03: Radiomics Track

**Status:** Not implemented. Phase 3.

**Depends on:** `config/`, `preprocessing/` (crops), `data/` (CaseTargets), `evaluation/`.

**Read also:** `SCHEMAS.md` (cached radiomics `.json`, `metrics.json`).

---

## Overview

Classical radiomics baseline: PyRadiomics feature extraction per (case, label)
→ feature matrix → classical ML (LightGBM or ElasticNet) with nested CV.

Kept deliberately simple in v1: no ComBat, no stability filters, no deep
methods. This is the baseline scatter has to beat.

---

## File: `src/scatterrad/models/radiomics/config.py`

PyRadiomics parameters per modality.

```python
def get_pyradiomics_config(modality: str) -> dict:
    """Returns IBSI-compliant config dict for pyradiomics."""
```

- **CT**: `binWidth=25`, `normalize=False` (already normalized upstream).
- **MR**: `binCount=32`, `normalize=True`, `normalizeScale=100`.

Enabled feature classes (both modalities):
- `firstorder`, `shape`, `glcm`, `glrlm`, `glszm`, `gldm`, `ngtdm`

Enabled image types:
- `Original`
- `Wavelet` (all sub-bands)
- `LoG` with `sigma=[1.0, 2.0, 3.0]` (in mm)

Return a plain dict — PyRadiomics expects YAML-style dicts.

Also define `def config_hash(cfg: dict) -> str` returning a short stable hash.

---

## File: `src/scatterrad/models/radiomics/extractor.py`

### Public API

```python
def extract_case_label(
    crop_path: Path,
    output_path: Path,
    modality: str,
    force: bool = False,
) -> dict[str, float]:
    """Load crop .npz, run pyradiomics, write cached .json, return features."""

def extract_all(
    paths: ScatterRadPaths,
    modality: str,
    num_workers: int = 0,
    force: bool = False,
) -> None:
    """Run extraction for every crop in preprocessed/crops/."""

def load_features(path: Path, expected_config_hash: str) -> dict[str, float] | None:
    """Load cached features, returning None if config hash mismatches."""
```

### Implementation notes

- PyRadiomics doesn't accept numpy arrays directly — wrap `image`/`mask`
  numpy into `SimpleITK.Image` first (use 1mm isotropic metadata on the crop;
  doesn't matter, features are computed from the array).
- Mask must be a binary ITK image; pyradiomics complains if label id != 1.
  Pass `label=1` and ensure mask is uint8 0/1.
- Cached `.json` includes `config_hash`; if user changes PyRadiomics config
  later, old caches are invalidated.
- Parallel: use `multiprocessing.Pool`. Radiomics extraction is CPU-heavy
  (~1–10 seconds per crop).
- Drop features that PyRadiomics returns with prefix `diagnostics_` — they're
  metadata, not features.

### Error cases

- Crop mask is empty → skip with warning (can happen if preprocessing wrote
  an empty mask, which shouldn't, but defensive).
- PyRadiomics raises → log error, skip case, don't crash the batch.
- Return dict has NaN/inf → replace with NaN, let trainer handle.

### Tests

- Mock PyRadiomics call, verify caching and hash invalidation logic.
- Integration test with a real tiny synthetic crop.

---

## File: `src/scatterrad/models/radiomics/trainer.py`

### Public API

```python
def train(
    paths: ScatterRadPaths,
    task: TaskConfig,
    dataset: DatasetConfig,
    schema: TargetsSchema,
    plans: PlansConfig,
    fold: int | None = None,
) -> None:
    """Run nested CV training. fold=None runs all folds; else runs just that one."""
```

Writes per-fold artifacts to `paths.result_dir(task.name, "radiomics", fold)`:
- `model.joblib`
- `metrics.json`
- `predictions.csv` (per-sample predictions on val fold)
- `features_used.txt` (which features survived filtering, in case of ElasticNet
  pre-filter)

### Feature matrix construction

```python
def build_feature_matrix(
    paths: ScatterRadPaths,
    task: TaskConfig,
    schema: TargetsSchema,
    labels: tuple[int, ...],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Returns (X, y, sample_ids).

    X: rows = samples, columns = features.
    y: target values (float for regression, int for classification).
    sample_ids: "<basename>__label<K>" for per_label, "<basename>" for per_case.
    """
```

For **per_label** targets:
- One row per (case, label) pair.
- Columns = feature names.
- `y[i]` = target value for that (case, label).
- Rows with NaN target dropped.

For **per_case** targets:
- One row per case.
- Columns = concatenation of all per-label features, prefixed with label id:
  `label020_original_firstorder_Mean`, etc.
- Missing labels for a case → that block is NaN.
- `y[i]` = per-case target value.
- Rows with NaN target dropped.

### Pipeline per outer fold

```python
1. Load splits.json, pick outer fold.
2. Build X_train, y_train, X_val, y_val.
3. Preprocess:
   - Drop columns with >= nan_threshold NaN fraction (default 0.1).
   - Impute remaining NaNs with median (fit on train only).
   - StandardScaler fit on train, transform both.
   - Drop columns with variance < variance_threshold (default 0.0).
4. Inner CV (5-fold on X_train) for hyperparam search with Optuna:
   - classifier="lightgbm":
       search num_leaves, max_depth, learning_rate, min_child_samples,
       reg_alpha, reg_lambda.
       objective = binary/multiclass/regression per target type.
       class weight = "balanced" for classification.
   - classifier="elasticnet":
       search alpha, l1_ratio.
       logistic regression for classification, linear for regression.
5. Refit best model on full X_train.
6. Predict on X_val.
7. Compute metrics (see evaluation module).
8. Save artifacts.
```

### LightGBM specifics

- For classification, use `LGBMClassifier` with `class_weight="balanced"`.
- For regression, `LGBMRegressor`.
- Set `n_estimators=2000` with early stopping on inner validation split
  (patience 50).
- Verbose=-1.

### ElasticNet specifics

- Scaling required. Already done in step 3.
- For binary classification: `LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=..., C=...)`.
- For multiclass: same, `multi_class='multinomial'`.
- For regression: `ElasticNet(alpha=..., l1_ratio=...)`.
- Store non-zero feature coefficients in `features_used.txt`.

### Optuna budget

`task.model_config.optuna_trials` (default 30). Use `TPESampler`, no pruning
for inner CV (it's fast anyway).

### Tests

- Build feature matrix: per_label and per_case shapes match expectations.
- Training on synthetic data with injected signal → model learns it.
- Determinism: same seed → same metrics.
- Missing label handling: per_case features with a missing label get NaN'd
  and imputed correctly.

---

## File: `src/scatterrad/models/radiomics/predictor.py`

### Public API

```python
def predict(
    paths: ScatterRadPaths,
    task_name: str,
    model_kind: str,
    inputs: list[Path],  # directory of new NIFTIs or single file
    fold: int | None = None,  # None = ensemble over all folds
) -> pd.DataFrame:
    """Run inference. Returns DataFrame with predictions."""
```

Applies preprocessing + extraction to new inputs (segmentation must already be
available or computed by nnunet externally), then applies the trained model.

Kept as a Phase 7 deliverable — don't build until Phase 3 (training) is solid.

---

## Metrics reported

For classification:
- AUC-ROC (binary) or macro-AUC (multiclass)
- Balanced accuracy
- F1 (macro for multiclass)
- Confusion matrix
- Precision/recall per class

For regression:
- MAE
- RMSE
- R²
- Pearson correlation

See `evaluation/metrics.py` for the implementation — trainer just calls it.

---

## Invariants

1. Radiomics features are cached; rerunning `train` doesn't recompute them.
2. Changing PyRadiomics config invalidates all caches (via config_hash).
3. StandardScaler, imputer, and feature selector are all fit on train only,
   never on val.
4. Inner CV is inside the outer train fold only.
5. Predictions are reproducible from `model.joblib` + seed.

---

## What NOT to do

- Do not filter features by univariate statistics on the full dataset
  (leakage).
- Do not fit hyperparams on outer-fold val (that's what inner CV is for).
- Do not skip the per-modality PyRadiomics config (CT and MR need different
  discretization — DEC004).
- Do not add ComBat, SMOTE, or stability filters. V2.
