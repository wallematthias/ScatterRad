# Module 02: Preprocessing

**Status:** Not implemented. Phases 1–2.

**Read also:** `SCHEMAS.md` (plans.json, splits.json, crop .npz), `DEC004_normalization_strategy.md`.

**Depends on:** `config/`, `paths`, `utils/`.

**Used by:** `data/`, `models/radiomics/`, `models/scatter/`.

---

## Overview

Preprocessing has two stages, both runnable once per dataset:

1. **Planner** (`scatterrad plan`) — reads raw data headers + samples voxels,
   derives a `plans.json`.
2. **Preprocessor** (`scatterrad preprocess`) — applies the plan to every
   training case, produces per-label `.npz` crops + cached radiomics features
   + patient-level CV splits.

Subsequent training runs reuse these artifacts. The planner + preprocessor are
NOT run per task.

---

## Stage 1: Planner

### File: `src/scatterrad/preprocessing/planner.py`

### Public API

```python
def plan(paths: ScatterRadPaths) -> PlansConfig:
    """Read raw data, derive plans, write plans.json. Returns the plan."""
```

### Inputs read

- `dataset.json` — modality, labels, num_training
- `imagesTr/*_0000.nii.gz` — headers only (SimpleITK metadata), plus a voxel
  sample for intensity stats
- `labelsTr/*.nii.gz` — full load per case to compute bboxes
- `targetsTr/*.json` — only checked for existence (label coverage)

### Outputs written

- `plans.json` (schema in `SCHEMAS.md`)

### Algorithm

1. For each training case:
   - Read image header → get spacing.
   - Load label volume → for each foreground label id present, compute bbox
     in voxel coords + in mm (using spacing).
   - For intensity stats (CT only): sample up to N=10000 foreground voxels
     from each case, accumulate.
2. Aggregate:
   - **target_spacing_mm** = median per-axis spacing across cases.
   - **bbox_percentiles** per label = p50, p95 of bbox sizes in mm, divided
     by target_spacing → voxel count.
   - **crop_size_voxels** = max over all labels of p95, rounded up to next
     multiple of 8, clamped to `[32, 128]` per axis.
   - **intensity_clip** (CT only): modality-aware default, configurable:
     - general CT: `[-1000, 1000]`
     - override via optional `scatterrad_config.json` or function arg (v1:
       hard-code `[-1000, 1000]`, add config later)
   - **intensity_mean/std** (CT only): computed from the sampled foreground
     voxels, after clipping. MR leaves these as `null` — per-case z-score is
     applied downstream.
   - **label_coverage**: for each foreground label id, count how many cases
     contain at least one voxel of that label.
3. Write `plans.json` via `PlansConfig.to_json()`.

### Error cases

- No training cases → raise, suggest checking `imagesTr/`.
- A label in `dataset.json` but present in 0 cases → warn (don't fail), include
  with `coverage=0` in plans.
- Inconsistent file endings → fail fast with a clear message.

### Performance

- Target: ≤1s per case on header-only pass (for up to ~5000 cases).
- Use `tqdm` for progress.
- No parallelism needed here; I/O-bound and fast.

---

### File: `src/scatterrad/config/plans.py`

### `PlansConfig` dataclass

```python
@dataclass(frozen=True)
class PlansConfig:
    version: int
    dataset_name: str
    modality: str
    target_spacing_mm: tuple[float, float, float]
    crop_size_voxels: tuple[int, int, int]
    crop_margin_mm: float
    intensity_clip: tuple[float, float] | None
    intensity_mean: float | None
    intensity_std: float | None
    orientation: str     # always "RAS" in v1
    label_coverage: dict[int, int]
    bbox_percentiles: dict[int, dict[str, tuple[int, int, int]]]

    @classmethod
    def from_json(cls, path: Path) -> "PlansConfig": ...

    def to_json(self, path: Path) -> None: ...
```

Re-export from `config/__init__.py`.

### Tests

- Valid round-trip (write → load → equal).
- Missing fields → `PlansConfigError`.
- Schema version mismatch → clear error with migration hint.

---

## Stage 2: Preprocessor (runner + helpers)

### Files

- `src/scatterrad/preprocessing/resample.py`
- `src/scatterrad/preprocessing/normalize.py`
- `src/scatterrad/preprocessing/crop.py`
- `src/scatterrad/preprocessing/splits.py`
- `src/scatterrad/preprocessing/runner.py` — orchestrates

### resample.py

Public: `def resample_to_spacing(image: sitk.Image, target_spacing: tuple[float, float, float], is_label: bool) -> sitk.Image`

- Nearest-neighbor for labels, B-spline order 3 for images.
- Preserve orientation. Reorient to RAS before resampling.
- Output metadata (origin, direction) preserved.

### normalize.py

Public: `def normalize(image: np.ndarray, mask: np.ndarray, plans: PlansConfig) -> np.ndarray`

- CT: clip to `intensity_clip` → z-score with `(intensity_mean, intensity_std)`.
- MR: compute per-case mean/std inside `mask`, z-score.
- Return float32.
- Clip + normalize is modality-branched per DEC004.
- Mask is the foreground (any non-zero label), not the per-label mask.

### crop.py

Public:
```python
def bbox_from_label(label_volume: np.ndarray, label_id: int) -> tuple[slice, slice, slice] | None:
    """None if label not present."""

def crop_around_bbox(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[slice, slice, slice],
    target_size: tuple[int, int, int],
    margin_voxels: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cropped_image, cropped_mask), both with shape == target_size.
    Pads with zero where crop extends beyond volume.
    Raises if target_size is smaller than bbox+margin (planner should prevent this)."""

def write_crop(path: Path, image: np.ndarray, mask: np.ndarray, meta: dict) -> None: ...

def read_crop(path: Path) -> tuple[np.ndarray, np.ndarray, dict]: ...
```

Centering rule: place bbox center at crop center; pad symmetrically.

### splits.py

Public:
```python
def generate_splits(
    basenames: list[str],
    targets: dict[str, CaseTargets],
    schema: TargetsSchema,
    n_folds: int = 5,
    seed: int = 42,
    stratification_target: str | None = None,
) -> list[dict[str, list[str]]]:
    """Deterministic patient-level stratified k-fold."""
```

Stratification rules:
- If `stratification_target` given and target is classification: stratify on
  majority class value.
- If per_label: stratify on "any positive" for binary targets.
- If regression: bin into quartiles and stratify on bin.
- If none given: plain k-fold.

### runner.py

Public: `def preprocess(paths: ScatterRadPaths, num_workers: int = 0) -> None`

Pipeline per case:
1. Load image + label via SimpleITK.
2. Reorient to RAS.
3. Resample image (B-spline) + label (NN) to `plans.target_spacing_mm`.
4. Convert to numpy.
5. Normalize image intensity (normalize.py).
6. For each foreground label id present:
   - Compute bbox.
   - Crop image + per-label mask (mask = `(label_volume == id).astype(uint8)`)
     to `plans.crop_size_voxels`, padding with 0.
   - Write `.npz` with `image`, `mask`, `meta`.
7. Log progress with tqdm.

Then once per dataset:
- Generate splits (default: no stratification, user can override via
  `--stratify <target_name>` CLI arg if we add it; v1: auto-stratify on first
  classification target, else plain kfold).
- Write `splits.json`.

Radiomics extraction is triggered here as well but lives in
`models/radiomics/extractor.py` (called from runner). See module 03.

Parallelism: process-pool over cases (`multiprocessing.Pool`), not threads
(SimpleITK releases GIL but pyradiomics doesn't). Radiomics can reuse the same
pool. Default `num_workers=0` → serial, for debuggability.

### Error cases

- Image/label shape mismatch after resample → hard fail, show spacings.
- Case has no foreground labels → warn, skip, don't write any crops.
- Output dir already populated → skip existing crops (idempotent).
- `plans.json` doesn't exist → fail with "run `scatterrad plan` first".

### Tests

- Unit: resample preserves orientation, label NN doesn't interpolate.
- Unit: normalize CT vs MR branches on synthetic arrays.
- Unit: crop padding correctness (bbox at corner, at edge, exact fit).
- Unit: splits are deterministic, patient-level, covers all cases exactly once.
- Integration: 3-case synthetic dataset, full pipeline → expected file
  structure.

---

## Invariants (enforced by tests)

1. Preprocessing is deterministic given the same inputs.
2. Rerunning `preprocess` is idempotent (skips existing crops).
3. Crop shape == `plans.crop_size_voxels` for every crop, always.
4. Splits are patient-level and cover every case exactly once per fold cycle.
5. No case appears in both train and val of the same fold.

---

## Open questions for implementer

- Default `crop_margin_mm = 10` — confirm with user after Phase 1 runs on real
  data. 10mm works for vertebrae, might be too little for lungs or too much
  for lymph nodes.
- N4 bias correction for MR — deferred to V2. Document the decision in-line.

## What NOT to do

- Do not do 2D processing. All volumes are 3D.
- Do not use `torchio` — SimpleITK is sufficient and adds no extra dep.
- Do not auto-retry on failures — let them surface clearly.
- Do not do per-case z-score on CT (DEC004).
