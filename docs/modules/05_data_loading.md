# Module 05: Data Loading

**Status:** Not implemented. Phase 5 (bundled with scatter training).

**Depends on:** `config/`, `paths`, `preprocessing/` (for reading crops).

**Used by:** `models/scatter/`.

---

## Overview

PyTorch `Dataset` and `Sampler` classes that yield (image, mask, target, meta)
tuples from preprocessed crops. One class handles both `per_label` and
`per_case` scopes via a single `scope` parameter.

The radiomics track does NOT use this module — it loads cached features
directly into pandas.

---

## File: `src/scatterrad/data/dataset.py`

### Public API

```python
class ScatterRadDataset(Dataset):
    def __init__(
        self,
        paths: ScatterRadPaths,
        basenames: list[str],
        schema: TargetsSchema,
        task: TaskConfig,
        plans: PlansConfig,
        labels: tuple[int, ...],
        augment: bool = False,
        use_scatter_cache: bool = False,
    ): ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict: ...
```

### Output format

**per_label scope** — one sample per (case, label). `__len__` = sum over cases
of labels present.

```python
{
    "image":    torch.Tensor [1, D, H, W],
    "mask":     torch.Tensor [1, D, H, W],
    "target":   torch.Tensor [] (scalar),
    "present":  torch.Tensor [] bool (always True, for API uniformity),
    "meta":     {"basename": str, "label_id": int},
}
```

**per_case scope** — one sample per case. `__len__` = number of cases.

```python
{
    "images":   torch.Tensor [N_labels, 1, D, H, W],
    "masks":    torch.Tensor [N_labels, 1, D, H, W],
    "target":   torch.Tensor [] (scalar),
    "present":  torch.Tensor [N_labels] bool,
    "meta":     {"basename": str, "label_ids": list[int]},
}
```

Missing labels: the corresponding slot is a zero-tensor and `present[l] = False`.

### Sample enumeration

At `__init__` time, build a list of tuples:
- per_label: `[(basename, label_id), ...]` filtered to cases with non-NaN
  target for that label.
- per_case: `[basename, ...]` filtered to cases with non-NaN per-case target.

Rejection rule (per_case): if a case has zero labels present in crops, drop
it. If some labels missing, keep with `present` mask.

### Crop loading

Use `scatterrad.preprocessing.crop.read_crop`. If `use_scatter_cache=True`,
check for a cached `.npy` first and load scatter coefficients directly.

### Augmentation

Applied only if `augment=True` and only during training. See augment.py.

---

## File: `src/scatterrad/data/augment.py`

### Public API

```python
def augment_crop(
    image: np.ndarray,   # [D, H, W]
    mask: np.ndarray,    # [D, H, W]
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]: ...
```

Applies in order, each with independent probability 0.5:
- Random flip along each axis (3 flips, each independent).
- Small rotation around a random axis: ±10° (use `scipy.ndimage.rotate` with
  `reshape=False`, order=3 for image, order=0 for mask).
- Small scale: ±5% (zoom + crop/pad back to original shape).

Do NOT use elastic deformation (see module 04).

Mask augmentation must mirror image augmentation exactly. Apply the same
rotation matrix to both.

### Tests

- Flips are their own inverse.
- Rotation preserves shape.
- Mask and image stay aligned after each augmentation (test with a known-pattern
  mask).

---

## File: `src/scatterrad/data/sampler.py`

### Public API

```python
class ClassBalancedSampler(Sampler):
    def __init__(
        self,
        dataset: ScatterRadDataset,
        num_samples: int | None = None,
        seed: int = 42,
    ): ...

    def __iter__(self): ...
    def __len__(self) -> int: ...
```

For classification targets: weight each sample inversely by its class
frequency in the epoch. For regression: uniform sampling (no balancing).

Applied only on the training loader, not val. Default `num_samples` = dataset
length.

### Tests

- Over many draws, class frequencies converge to ~uniform.
- Deterministic given seed.
- Regression target → falls back to uniform.

---

## Collation

Default PyTorch collate works for per_label. For per_case, `N_labels` is
constant (determined by `task.resolved_labels(schema)`) so default collate
works there too.

If a case is missing a label, the slot is still present as a zero tensor.
This keeps the batch tensor rectangular at the cost of some wasted compute —
acceptable given the rest of the architecture.

No custom `collate_fn` needed in v1.

---

## Invariants

1. Iteration order is deterministic for a given seed.
2. Samples are patient-level independent — no crop appears in both train
   and val within a fold.
3. NaN target values never reach the model (filtered at `__init__`).
4. Crop tensors returned are always `plans.crop_size_voxels` shape.

---

## What NOT to do

- Do not do augmentation in `__getitem__` for val (only train).
- Do not apply augmentation to scatter-cached features (they're already
  computed; augmentation would break them). If augmentation is on,
  `use_scatter_cache` must be False — enforce at `__init__` with an error.
- Do not use DataLoader's `num_workers > 0` without testing — numpy file I/O
  can deadlock with fork. Prefer `num_workers=2` as a conservative default.
