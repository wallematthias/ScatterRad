# SCHEMAS.md — All JSON contracts

Single source of truth for every JSON file ScatterRad reads or writes.
Parsers live in `src/scatterrad/config/` and enforce these schemas.

---

## 1. `dataset.json` (raw input, nnunet-compatible)

Location: `ScatterRad_raw/<Dataset>/dataset.json`

```json
{
  "name": "SpineVertebraeCT",
  "modality": {"0": "CT"},
  "labels": {
    "0": "background",
    "20": "L1", "21": "L2", "22": "L3", "23": "L4", "24": "L5"
  },
  "numTraining": 312,
  "file_ending": ".nii.gz"
}
```

**Constraints (enforced by parser):**
- `modality` has exactly one entry with key `"0"`. (V1 is single-modality; see
  DEC001.)
- `modality[0]` ∈ {`"CT"`, `"MR"`, `"MRI"`}. `MRI` → canonicalized to `MR`.
- `labels` has at least one foreground label.
- Background is either the label explicitly named `"background"` or id `0`.
- Label ids must parse as ints.

Parser: `scatterrad.config.load_dataset_config` → `DatasetConfig`.

---

## 2. `targets.json` (raw input, task schema)

Location: `ScatterRad_raw/<Dataset>/targets.json`

Defines what can be predicted from this dataset.

```json
{
  "fracture": {
    "type": "classification",
    "scope": "per_label",
    "num_classes": 2,
    "applicable_labels": [20, 21, 22, 23, 24]
  },
  "metastasis": {
    "type": "classification",
    "scope": "per_label",
    "num_classes": 2,
    "applicable_labels": [20, 21, 22, 23, 24]
  },
  "age": {
    "type": "regression",
    "scope": "per_case",
    "applicable_labels": [20, 21, 22, 23, 24]
  },
  "sex": {
    "type": "classification",
    "scope": "per_case",
    "num_classes": 2,
    "applicable_labels": [20, 21, 22, 23, 24]
  }
}
```

**Per-target fields:**

| Field               | Type       | Required             | Notes                                 |
| ------------------- | ---------- | -------------------- | ------------------------------------- |
| `type`              | str        | yes                  | `classification` or `regression`      |
| `scope`             | str        | yes                  | `per_label` or `per_case`             |
| `num_classes`       | int ≥ 2    | if classification    | must be absent for regression         |
| `applicable_labels` | list[int]  | required for per_label; optional for per_case | subset of dataset.labels |

Parser: `scatterrad.config.load_targets_schema` → `TargetsSchema`.

Cross-validates against `dataset.json` label ids when both are loaded.

---

## 3. `targetsTr/<basename>.json` (raw input, per-case ground truth)

Location: `ScatterRad_raw/<Dataset>/targetsTr/<basename>.json`

One file per training case, same basename as image/label. All keys optional —
missing = NaN.

```json
{
  "age": 45,
  "sex": 0,
  "fracture":   {"20": 1, "21": 0, "22": 0, "23": 0, "24": 0},
  "metastasis": {"20": 0, "21": 1, "22": 0, "23": 0, "24": 0}
}
```

**Rules:**
- `per_case` targets: scalar (int or float).
- `per_label` targets: object `{label_id (string): value}`.
- Keys not present in `targets.json` are silently ignored (lets users store
  extra metadata).
- Label ids outside `applicable_labels` are silently ignored.
- Missing values can be omitted entirely, or set to `null`.

Parser: `scatterrad.config.load_case_targets` → `CaseTargets`.

---

## 4. `task.json` (user input, experiment spec)

Location: anywhere the user keeps it. Passed explicitly to `scatterrad train`.

**Minimal:**
```json
{
  "name": "metastasis_scatter_v1",
  "target": "metastasis",
  "model": "scatter"
}
```

**Full:**
```json
{
  "name": "metastasis_scatter_v1",
  "target": "metastasis",
  "labels": [20, 21, 22, 23, 24],
  "model": "scatter",
  "cv": {"folds": 5, "seed": 42},
  "model_config": {
    "J": 2,
    "L": 2,
    "crop_size": [64, 64, 64],
    "mask_mode": "zero"
  }
}
```

**Fields:**

| Field          | Type       | Required | Default  | Notes                                        |
| -------------- | ---------- | -------- | -------- | -------------------------------------------- |
| `name`         | str        | yes      |          | `[A-Za-z0-9_-]+` — used as dirname           |
| `target`       | str        | yes      |          | must exist in targets.json                   |
| `model`        | str        | yes      |          | `radiomics` or `scatter`                     |
| `labels`       | list[int]  | no       | schema's applicable_labels | overrides schema       |
| `cv.folds`     | int ≥ 2    | no       | 5        |                                              |
| `cv.seed`      | int        | no       | 42       |                                              |
| `model_config` | object     | no       | `{}`     | forwarded to the model class                 |

**Model-specific `model_config` keys:**

*radiomics:*
- `classifier`: `"lightgbm"` (default) \| `"elasticnet"`
- `feature_filter.variance_threshold`: float (default 0.0)
- `feature_filter.nan_threshold`: float (default 0.1)
- `optuna_trials`: int (default 30)

*scatter:*
- `J`: int, number of scales (default 2)
- `L`: int, angular resolution (default 2)
- `crop_size`: [D, H, W] (default from plans.json)
- `mask_mode`: `"zero"` (default) \| `"post_pool"` (see DEC003)
- `conv_channels`: int (default 32)
- `dropout`: float (default 0.3)
- `epochs`: int (default 100)
- `batch_size`: int (default 16)
- `lr`: float (default 1e-3)
- `weight_decay`: float (default 1e-4)
- `augment`: bool (default true) — flips + mild affine
- `cache_scatter_output`: bool (default true) — see DEC005

Parser: `scatterrad.config.load_task_config` → `TaskConfig`.

---

## 5. `plans.json` (generated by planner)

Location: `ScatterRad_preprocessed/<Dataset>/plans.json`

```json
{
  "version": 1,
  "dataset_name": "SpineVertebraeCT",
  "modality": "CT",
  "target_spacing_mm": [1.0, 1.0, 1.0],
  "crop_size_voxels": [64, 64, 64],
  "crop_margin_mm": 10.0,
  "intensity_clip": [-1000, 1000],
  "intensity_mean": -150.3,
  "intensity_std": 280.1,
  "orientation": "RAS",
  "label_coverage": {
    "20": 310, "21": 310, "22": 309, "23": 308, "24": 305
  },
  "bbox_percentiles": {
    "20": {"p50": [40, 42, 38], "p95": [58, 60, 54]},
    "21": {"p50": [41, 43, 39], "p95": [59, 61, 55]}
  }
}
```

**Fields:**

| Field               | Type       | Notes                                             |
| ------------------- | ---------- | ------------------------------------------------- |
| `version`           | int        | schema version, for future migrations             |
| `target_spacing_mm` | [z, y, x]  | median of training set (per-axis)                 |
| `crop_size_voxels`  | [D, H, W]  | 95th percentile of per-label bbox + margin, rounded up to multiple of 8, clamped to [32, 128] |
| `crop_margin_mm`    | float      | extra margin beyond bbox                          |
| `intensity_clip`    | [lo, hi]   | HU clip (CT only; `null` for MR)                  |
| `intensity_mean`    | float      | dataset mean over foreground (CT) or `null` for MR per-case z-score |
| `intensity_std`     | float      | dataset std over foreground (CT) or `null`        |
| `label_coverage`    | dict       | per-label count of cases containing that label    |
| `bbox_percentiles`  | dict       | per-label bbox size percentiles (for debugging)   |

Writer: `scatterrad.preprocessing.planner.plan` (to be implemented).

---

## 6. `splits.json` (generated by preprocessor)

Location: `ScatterRad_preprocessed/<Dataset>/splits.json`

```json
{
  "seed": 42,
  "n_folds": 5,
  "strategy": "patient_stratified_kfold",
  "stratification_key": "has_any_metastasis",
  "folds": [
    {"train": ["case001", "case003", ...], "val": ["case002", "case005", ...]},
    {"train": [...], "val": [...]},
    ...
  ]
}
```

Splits are patient-level (one case = one patient) and generated deterministically
from the seed. Reused across all tasks on the same dataset.

Writer: `scatterrad.preprocessing.runner.generate_splits`.

---

## 7. Per-crop sidecar `<basename>_label<K>.npz`

Not JSON but documented here for completeness. Contents:

```python
{
    "image": np.ndarray,         # float32, shape = plans.crop_size
    "mask":  np.ndarray,         # uint8,   shape = plans.crop_size, 1 where label, 0 elsewhere
    "meta":  np.ndarray,         # 0-d object array wrapping a dict with:
                                 #   basename, label_id, original_spacing_mm,
                                 #   crop_origin_voxels, bbox_voxels
}
```

Writer: `scatterrad.preprocessing.crop.write_crop`.
Reader: `scatterrad.data.dataset.load_crop`.

---

## 8. Cached radiomics `<basename>_label<K>.json`

Output of PyRadiomics feature extraction, saved raw.

```json
{
  "pyradiomics_version": "3.1.0",
  "config_hash": "a1b2c3d4",
  "features": {
    "original_firstorder_Mean": 123.4,
    "original_glcm_Contrast": 5.67,
    "wavelet-LLH_firstorder_Entropy": 2.34,
    ...
  }
}
```

`config_hash` is a hash of the PyRadiomics config dict — if the config changes,
cached files are invalidated on next load.

Writer/Reader: `scatterrad.models.radiomics.extractor`.

---

## 9. `metrics.json` (per-fold training output)

Location: `ScatterRad_results/<Dataset>/<task>__<model>__fold<F>/metrics.json`

```json
{
  "task": "metastasis_scatter_v1",
  "model": "scatter",
  "fold": 0,
  "n_train": 248,
  "n_val": 62,
  "target_type": "classification",
  "metrics": {
    "auc": 0.812,
    "balanced_accuracy": 0.74,
    "f1": 0.71,
    "confusion_matrix": [[42, 8], [10, 2]]
  },
  "attention_weights_mean": {"20": 0.15, "21": 0.42, "22": 0.18, "23": 0.14, "24": 0.11},
  "runtime_seconds": 312.5
}
```

`attention_weights_mean` is populated only for scatter + per_case targets.

Writer: `scatterrad.evaluation.report`.
