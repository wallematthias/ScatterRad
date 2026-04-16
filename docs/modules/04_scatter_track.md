# Module 04: Scatter Track

**Status:** Not implemented. Phases 4–5.

**Depends on:** `config/`, `data/`, `evaluation/`.

**Read also:** `DEC002_attention_pool_default.md`, `DEC003_mask_mode_configurable.md`, `DEC005_planner_caches_scatter.md`.

---

## Overview

Fixed front-end (3D harmonic scattering) + learnable back-end (shallow 3D
CNN) + attention pooling across labels (for per_case targets) + MLP head.

Designed for small-data regimes: most parameters are in the fixed front-end
(which has zero learnables). Learnable part is ~20k params total.

---

## Architecture

```
    [B, 1, D, H, W]  (per-label crop, mask-applied)
         │
         ▼
    HarmonicScattering3D (fixed, no grad)
         │
    [B, C_s, D', H', W']  where D'=D/2^J
         │
         ▼
    BatchNorm3d
    Conv3d(C_s → 32, 3×3×3, pad=1) + ReLU + BN
    Conv3d(32  → 32, 3×3×3, pad=1) + ReLU + BN
         │
    [B, 32, D', H', W']
         │
         ▼
    Masked global average pool (mask downsampled to D')
         │
    [B, 32]   ← per-label feature vector f_l
         │
         ├── per_label target:
         │     head(f_l) → logits
         │
         └── per_case target (variable labels per case):
               stack f_l for l in case.labels  →  [B, N_labels, 32]
               MaskedAttentionPool              →  [B, 32] + weights
               head(pooled) → logits
```

---

## File: `src/scatterrad/models/scatter/frontend.py`

### Public API

```python
class ScatterFrontend(nn.Module):
    def __init__(
        self,
        crop_size: tuple[int, int, int],  # (D, H, W)
        J: int = 2,
        L: int = 2,
        integral_powers: list[float] = (0.5, 1.0, 2.0),
        mask_mode: str = "zero",  # "zero" | "post_pool"
    ): ...

    @property
    def out_channels(self) -> int: ...

    @property
    def out_shape(self) -> tuple[int, int, int]: ...

    def forward(
        self,
        image: torch.Tensor,   # [B, 1, D, H, W]
        mask: torch.Tensor,    # [B, 1, D, H, W], uint8/bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (scatter_output, downsampled_mask).

        scatter_output: [B, C_s, D', H', W']
        downsampled_mask: [B, 1, D', H', W']  — for downstream masked pooling
        """
```

### Mask modes

- `"zero"` (default): multiply image by mask before scattering. Scatter
  "sees" only voxels inside the label. Downsampled mask still returned for
  consistency (all-ones where the downsampled mask would be one).
- `"post_pool"`: scatter the full image (no masking). Downsampled mask is
  returned so the backend can do masked GAP.

See DEC003 for rationale.

### Kymatio specifics

```python
from kymatio.torch import HarmonicScattering3D

self.scattering = HarmonicScattering3D(
    J=J, L=L,
    shape=crop_size,
    sigma_0=1,
    integral_powers=integral_powers,
)
```

- Output shape per kymatio: `[B, C_s, D/2^J, H/2^J, W/2^J]` where C_s depends
  on J, L, and `len(integral_powers)`. Compute at `__init__` time with a
  dummy forward pass to populate `self.out_channels` and `self.out_shape`.
- `requires_grad_(False)` on all kymatio parameters — they're fixed.
- Move to device at model construction.

### Mask downsampling

Use `F.avg_pool3d(mask.float(), kernel_size=2**J, stride=2**J)` → binarize at
threshold 0.5 (preserves interior, shrinks edges). This is an approximation
but adequate for the pooling use.

### Tests

- Forward pass shape with small input (16³, J=1, L=1).
- Both mask modes produce outputs of the same shape.
- No gradients flow through scattering.

---

## File: `src/scatterrad/models/scatter/backend.py`

### Public API

```python
class ScatterBackend(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
    ): ...

    def forward(
        self,
        x: torch.Tensor,         # [B, C_s, D', H', W']
        mask: torch.Tensor,      # [B, 1, D', H', W']
    ) -> torch.Tensor:
        """Masked global average pool output: [B, hidden_channels]."""
```

Two conv blocks:

```
BatchNorm3d(in_channels)
Conv3d(in_channels → hidden, 3, pad=1) + ReLU + BN
Conv3d(hidden → hidden, 3, pad=1) + ReLU + BN
```

Then masked GAP:

```python
def masked_gap(feat, mask):
    # feat: [B, H, D, H, W]; mask: [B, 1, D, H, W]
    denom = mask.sum(dim=(2, 3, 4)).clamp(min=1.0)
    pooled = (feat * mask).sum(dim=(2, 3, 4)) / denom
    return pooled
```

### Tests

- Forward pass correct shape.
- Masked GAP with all-ones mask == regular GAP.
- Masked GAP ignores zero regions.

---

## File: `src/scatterrad/models/scatter/pooling.py`

### Public API

```python
class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int): ...

    def forward(
        self,
        feats: torch.Tensor,         # [B, N_labels, dim]
        present_mask: torch.Tensor,  # [B, N_labels] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled, weights).

        pooled: [B, dim]
        weights: [B, N_labels]  (softmax, zero where present_mask is False)
        """
```

Implementation:

```python
self.score = nn.Linear(dim, 1)
logits = self.score(feats).squeeze(-1)          # [B, N_labels]
logits = logits.masked_fill(~present_mask, -1e9)
weights = logits.softmax(dim=1)
pooled = (feats * weights.unsqueeze(-1)).sum(dim=1)
```

Weights returned so they can be logged for interpretability (see metrics.json).

### Tests

- Attention with all labels present == softmax-weighted sum.
- Masked label has weight ≈ 0.
- Gradient flows to present labels only.

---

## File: `src/scatterrad/models/scatter/model.py`

### Public API

```python
class ScatterRadModel(nn.Module):
    def __init__(
        self,
        crop_size: tuple[int, int, int],
        target_type: TargetType,
        target_scope: TargetScope,
        num_classes: int | None,         # for classification
        J: int = 2,
        L: int = 2,
        hidden_channels: int = 32,
        dropout: float = 0.3,
        mask_mode: str = "zero",
    ): ...

    def forward(self, batch: dict) -> dict:
        """Returns dict with 'logits' and optionally 'attention_weights'."""
```

### Batch shape conventions

- **per_label**: batch[`image`] = `[B, 1, D, H, W]`, batch[`mask`] same,
  batch[`target`] = `[B]`.
- **per_case**: batch[`images`] = `[B, N_labels, 1, D, H, W]`, batch[`masks`]
  = same, batch[`present`] = `[B, N_labels]` bool, batch[`target`] = `[B]`.
  Labels missing for a case have zero-tensor images and `present=False`.

The model inspects `target_scope` at init and dispatches internally. For
per_case, it runs frontend+backend on each label independently (flatten to
`[B*N, 1, D, H, W]`, forward, unflatten to `[B, N, hidden]`), then attention
pools.

### Output head

```python
out_dim = 1 if target_type is REGRESSION or num_classes == 2 else num_classes
self.head = nn.Sequential(
    nn.Linear(hidden_channels, hidden_channels),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_channels, out_dim),
)
```

### Tests

- Full forward pass both scopes, verified shapes.
- Parameter count: backend + pool + head, excluding kymatio.
- Gradients flow through backend, not frontend.

---

## File: `src/scatterrad/models/scatter/scatter_cache.py`

Optional on-disk cache of scattering outputs to skip the frontend during training.

### Public API

```python
def precompute_and_cache(
    paths: ScatterRadPaths,
    frontend: ScatterFrontend,
    device: str = "cpu",
) -> None:
    """Write cached scatter outputs to preprocessed/scatter_cache/<basename>_label<K>.npy."""

def load_cached_scatter(paths: ScatterRadPaths, basename: str, label_id: int) -> np.ndarray | None:
    """Returns None if cache miss."""
```

- Only valid for a specific `(J, L, mask_mode, crop_size)` tuple. Cache
  directory includes a hash of these in its name: `scatter_cache_<hash>/`.
- Computed once after preprocessing; subsequent training runs skip the front
  end entirely.
- See DEC005.

---

## File: `src/scatterrad/models/scatter/trainer.py`

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
    """Run CV training. fold=None → all folds; int → single fold."""
```

### Pipeline per fold

1. Build `ScatterRadDataset` for train and val splits (see module 05).
2. Instantiate model from `task.model_config`.
3. Optionally: precompute scatter cache (if `cache_scatter_output=True` and
   cache not present for this config).
4. Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`), configurable.
5. Loss by target type:
   - binary classification: `BCEWithLogitsLoss(pos_weight=...)` where pos_weight
     computed from train label frequencies
   - multiclass: `CrossEntropyLoss(weight=...)`
   - regression: `SmoothL1Loss()` (Huber)
6. Epochs: up to `task.model_config.epochs` (default 100), early stopping
   on val metric with patience 15.
7. Inner val split: 20% of outer train, stratified (same logic as splits.py).
8. Augment on-the-fly (see module 05): flips + affine, only if
   `augment=True`.
9. Per epoch:
   - train one epoch
   - validate → compute primary metric (AUC for binary, macro-AUC for
     multiclass, −MAE for regression)
   - if improved: save checkpoint
   - early stop if no improvement for patience
10. Save artifacts:
    - `checkpoint.pt` (best model weights + optimizer state + epoch)
    - `metrics.json`
    - `predictions.csv`
    - `log.txt` (per-epoch losses and metrics)

### Important training details

- Use `torch.cuda.amp` for mixed precision if GPU.
- Set seeds: `torch.manual_seed(seed)`, `np.random.seed(seed)`,
  `torch.backends.cudnn.deterministic = True`.
- Batch size defaults to 16 but should auto-fallback if OOM (halve and retry
  up to 3 times).
- Log attention weights averaged over val set, saved in metrics.json
  (per_case only).

### Tests

- Single training step runs without error on CPU with a 2-case synthetic
  dataset.
- Early stopping triggers when val metric plateaus.
- Deterministic with fixed seed.
- per_label and per_case scopes both runnable.

---

## What NOT to do

- Do not use elastic deformation augmentation (interacts poorly with
  scattering's deformation invariance guarantees).
- Do not train the scattering transform itself.
- Do not use dropout inside the conv blocks (small model, already regularized
  by BN + small capacity).
- Do not add skip connections. Two-block shallow design is intentional.
- Do not cache scatter outputs during training — cache happens during
  preprocessing only, to avoid per-epoch I/O churn.
