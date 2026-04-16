# DEC003: Mask treatment for scattering — both modes, zero-mask default

**Status:** Accepted (v1)
**Date:** 2026-04

## Decision

The scatter front-end supports two mask modes, with `"zero"` as the default:

- **`"zero"`**: multiply image by mask before scattering. Semantically matches
  PyRadiomics (features computed strictly inside ROI). Injects a sharp edge
  at the mask boundary which scattering wavelets will pick up as high-freq
  content.

- **`"post_pool"`**: scatter the full (mask-free) crop, then apply a
  downsampled mask at the masked global average pool. Preserves real
  anatomy at the boundary; includes surrounding tissue in scattering but not
  in pooling.

## Rationale

- **Zero-mask pros**: comparable to PyRadiomics (clean semantics, same
  ROI-only principle). Fewer confounders from surrounding tissue.
- **Zero-mask cons**: mask boundary introduces high-freq artifacts the
  scattering will describe. Some coefficients end up encoding mask shape,
  not label texture.
- **Post-pool pros**: no artificial boundaries. Scattering sees real tissue.
- **Post-pool cons**: boundary coefficients mix inside and outside info.
  Less ROI-pure.

Both are defensible. The empirical question — does boundary context help? —
is dataset-dependent. Supporting both as a config flag costs one
`if`-branch.

## Consequences

- `ScatterFrontend(..., mask_mode="zero"|"post_pool")` supports both.
- `task.json` → `model_config.mask_mode` controls it per experiment.
- Default is `"zero"` for comparability with the radiomics track.
- Unit tests cover both modes.

## V2 path

- Learnable boundary weighting (smooth transition from inside to outside).
- Boundary-only scattering features (explicit boundary descriptors).
