# DEC002: Masked attention pooling for per-case targets

**Status:** Accepted (v1)
**Date:** 2026-04

## Decision

For per_case targets aggregated across multiple labels (e.g., predicting
`age` from vertebrae [20–24]), the default aggregation is **masked attention
pooling** across per-label feature vectors.

## Rationale

Evaluated four options:

1. **Attention pool (chosen)** — learnable scalar score per label, softmax,
   weighted sum. Handles missing labels naturally. Weights are interpretable.
2. **Concat** — requires fixed label count; breaks on missing labels;
   imposes label ordering.
3. **Joint crop** — undermines per-label-texture philosophy.
4. **Mean/max pool** — simple but no interpretability; dominated by attention.

Attention adds only `~dim+1` params (one linear layer → scalar per label).
Negligible for <500 cases.

## Consequences

- `MaskedAttentionPool` in `models/scatter/pooling.py` is the default.
- Mean/max pool available as config option for ablations
  (`model_config.aggregation: "attention" | "mean" | "max"`). Phase 5 item.
- Weights are logged per-case in `metrics.json` for interpretability
  (mean across val set).

## V2 path

- Multi-head attention (across labels).
- Gated attention (Ilse et al., MIL).
- Cross-attention between labels (e.g., vertebra-vertebra relationships).
