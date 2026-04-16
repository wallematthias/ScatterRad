# DEC001: One modality per project

**Status:** Accepted (v1)
**Date:** 2026-04

## Decision

Each ScatterRad dataset contains exactly one modality. `dataset.json` must
have `modality: {"0": "CT"}` or `{"0": "MR"}`. Multi-channel input (e.g.,
T1+T2+FLAIR) is NOT supported in v1.

## Rationale

- Intensity normalization differs fundamentally between CT and MR. Branching
  on modality in every preprocessing step is clean; supporting both in one
  project creates combinatorial config complexity.
- PyRadiomics config (bin width vs bin count) differs by modality.
- Users typically want per-modality pipelines anyway.
- Follows nnunet's practice of one-dataset-one-task.

## Consequences

- Parsers reject multi-channel `modality` dicts with a clear error.
- Users wanting multi-modal fusion split their data into separate datasets
  and combine at feature level (radiomics) or as a V2 feature.

## V2 path

Multi-channel input can be added by extending `dataset.json` schema to allow
multiple modality entries, with per-channel normalization configs. Deferred.
