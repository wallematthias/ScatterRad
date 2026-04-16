# DEC005: Scatter output caching

**Status:** Accepted (v1)
**Date:** 2026-04

## Decision

Scattering transform outputs are cached to disk once per dataset per
`(J, L, mask_mode, crop_size)` tuple. During training, the front-end is
skipped and cached outputs are loaded directly.

Caching is controlled by `task.model_config.cache_scatter_output` (default
`True`). Caches live in:

```
preprocessed/scatter_cache_<hash>/<basename>_label<K>.npy
```

Where `<hash>` is an 8-character digest of `(J, L, mask_mode, crop_size)`.

## Rationale

Scattering is deterministic and input-deterministic. For fixed J, L, crop
size, and mask mode, the output for a given crop never changes. Recomputing
it every epoch is wasteful.

Trade-offs:
- **Pros**: massive training speedup (scattering is ~50–80% of forward-pass
  time for small CNN backends); removes frontend from hot path.
- **Cons**: disk usage grows. For 500 cases × 5 labels, J=2, 64³ crops,
  C_s≈80 channels at 16³ spatial → ~300KB per label × 2500 = ~750MB. Cheap.
- **Cons**: augmentation incompatible with caching (see module 05). If user
  wants augmentation, they must set `cache_scatter_output=False` OR disable
  augmentation (`augment=False`).

## Consequences

- `scatter_cache.py` module handles the cache (write + read).
- Cache is computed lazily on first training run for a config (not
  pre-emptively during preprocess — too much disk if user never runs
  scatter).
- `ScatterRadDataset` checks for cache before loading crops.
- Enforced mutual exclusion: `augment=True` + `use_scatter_cache=True` raises
  an error at Dataset init.

## V2 path

- Augmentation-aware caching (pre-compute N augmented versions per crop).
- Per-epoch cache warm-up on GPU for massive datasets.
- Option to offload cache to shared memory / mmap for multi-worker loading.
