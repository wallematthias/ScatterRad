# DEC004: Intensity normalization strategy

**Status:** Accepted (v1)
**Date:** 2026-04

## Decision

**CT:**
- Clip to `[-1000, 1000]` HU (configurable in future; hard-coded in v1).
- Z-score using **dataset-level** mean and std computed over foreground
  voxels across the training set.
- Do NOT per-case z-score.

**MR:**
- Per-case z-score within the foreground mask.
- No dataset-level stats.
- N4 bias correction NOT performed in v1 (V2).

**Radiomics (PyRadiomics config, separate from CNN input):**
- CT: `binWidth=25`, no PyRadiomics-internal normalization.
- MR: `binCount=32`, PyRadiomics-internal `normalize=True` with
  `normalizeScale=100`.

## Rationale

### CT: dataset-level z-score

CT has absolute physical meaning (HU values correspond to specific tissue
densities). Per-case z-score would destroy this, making bone-in-case-A
indistinguishable from soft-tissue-in-case-B.

Dataset-level z-score with fixed clipping is IBSI-compliant and standard in
PyRadiomics pipelines.

### MR: per-case z-score

MR intensity values have no absolute scale — they depend on scanner,
sequence, coil, and countless acquisition parameters. Per-case
normalization is necessary.

For cross-scanner robustness, Nyul histogram standardization is an option
but adds complexity. Deferred to V2.

### Radiomics discretization

PyRadiomics requires intensity discretization for texture features (GLCM,
GLRLM, etc.):
- Fixed bin width: suited to modalities with absolute scale (CT).
- Fixed bin count: suited to modalities with relative scale (MR).

These are IBSI recommendations.

## Consequences

- Planner computes CT dataset stats over foreground voxels; skips this for
  MR.
- `plans.json` has `intensity_clip`, `intensity_mean`, `intensity_std` for CT;
  these are `null` for MR.
- `normalize.py` branches on modality.
- Radiomics config also branches on modality (module 03).

## V2 path

- N4 bias correction pipeline for MR.
- Nyul histogram standardization for MR.
- Configurable CT clipping windows (lung, abdomen, bone, mediastinum).
- ComBat post-hoc harmonization for multi-scanner data.
