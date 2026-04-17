# IMPLEMENTATION_PLAN.md — Phased build order

**Read before starting any session.** Marks what's done and what's next.
Each phase = one coherent commit. Do not skip ahead. Do not combine phases.

---

## Phase 0 — Foundation (DONE, needs verification)

**Status:** code written, tests written, **never executed.**

Artifacts:
- `pyproject.toml`, `README.md`
- `src/scatterrad/__init__.py`, `paths.py`, `cli.py`
- `src/scatterrad/config/{__init__,dataset,targets,task}.py`
- `tests/test_paths.py`, `test_dataset_config.py`, `test_targets_config.py`,
  `test_task_config.py`, `test_cli_smoke.py`

**Verification task (do this first in your first session):**

```bash
pip install -e ".[dev]"
pytest tests/ -x
```

If any test fails, fix the code (not the test) unless the test is clearly
wrong. Do not proceed to Phase 1 until all tests pass.

---

## Phase 1 — Preprocessing planner (DONE)

**Spec:** `docs/modules/02_preprocessing.md` (section: planner)

Deliverables:
- `src/scatterrad/preprocessing/planner.py`
  - Public: `def plan(paths: ScatterRadPaths) -> PlansConfig`
- `src/scatterrad/config/plans.py` — `PlansConfig` dataclass + JSON I/O
- Wire `cmd_plan` in `cli.py` to call `plan()` and write `plans.json`
- Tests: `tests/test_planner.py`

Scope boundary:
- Planner reads raw images + labels, computes spacing/crop_size/norm stats,
  writes `plans.json`. Nothing else.
- Uses SimpleITK to read NIFTI headers (cheap) and a sample of voxels for
  intensity stats (not full loads).

**Exit criteria:**
- `scatterrad plan <dataset>` runs end-to-end on a toy 3-case dataset.
- `plans.json` matches schema in `SCHEMAS.md`.
- `pytest tests/test_planner.py` passes.

---

## Phase 2 — Preprocessing runner (crops + caching) (DONE)

**Spec:** `docs/modules/02_preprocessing.md` (sections: resample, normalize, crop)

Deliverables:
- `src/scatterrad/preprocessing/resample.py`
- `src/scatterrad/preprocessing/normalize.py`
- `src/scatterrad/preprocessing/crop.py`
- `src/scatterrad/preprocessing/runner.py` — orchestrates all three + writes
  `.npz` crops
- `src/scatterrad/preprocessing/splits.py` — patient-stratified k-fold
- Wire `cmd_preprocess` in `cli.py`
- Tests: one per module + integration test

**Exit criteria:**
- `scatterrad preprocess <dataset>` produces `crops/*.npz` and `splits.json`
  for a toy dataset.
- Crops are deterministic: rerunning produces byte-identical outputs.

---

## Phase 3 — Radiomics track (end-to-end baseline) (DONE)

**Spec:** `docs/modules/03_radiomics_track.md`

Deliverables:
- `src/scatterrad/models/radiomics/extractor.py` — pyradiomics wrapper with
  caching
- `src/scatterrad/models/radiomics/trainer.py` — feature matrix build, CV loop,
  classifier fit
- `src/scatterrad/models/radiomics/config.py` — PyRadiomics config per modality
- Wire `cmd_train` for `model == "radiomics"`
- Tests

**Exit criteria:**
- `scatterrad train <dataset> <task.json>` (radiomics model) produces
  `metrics.json` for each fold.
- Works for all four {per_label, per_case} × {classification, regression}
  combinations.

This is the **baseline** the scatter track must beat. Get it solid.

---

## Phase 4 — Scatter track: frontend + backend (DONE)

**Spec:** `docs/modules/04_scatter_track.md`

Deliverables:
- `src/scatterrad/models/scatter/frontend.py` — `HarmonicScattering3D` wrapper
  with mask-mode handling
- `src/scatterrad/models/scatter/backend.py` — shallow 3D conv stack
- `src/scatterrad/models/scatter/pooling.py` — `MaskedAttentionPool`
- `src/scatterrad/models/scatter/model.py` — assembles frontend + backend +
  pool + head
- Tests with small synthetic tensors

**Exit criteria:**
- Unit tests pass on CPU with tiny inputs (e.g., 16³ crops).
- Forward pass shape is correct for both per_label and per_case scenarios.
- No training loop yet.

---

## Phase 5 — Scatter track: data + training (DONE)

**Spec:** `docs/modules/04_scatter_track.md` (section: training) +
`docs/modules/05_data_loading.md`

Deliverables:
- `src/scatterrad/data/dataset.py` — `ScatterRadDataset` (PyTorch Dataset)
- `src/scatterrad/data/sampler.py` — class-balanced sampler
- `src/scatterrad/data/augment.py` — flips + mild affine
- `src/scatterrad/models/scatter/trainer.py` — training loop with early
  stopping, CV
- `src/scatterrad/models/scatter/scatter_cache.py` — optional on-disk cache
  of scattering outputs
- Wire `cmd_train` for `model == "scatter"`

**Exit criteria:**
- `scatterrad train <dataset> <task.json>` (scatter model) produces
  `checkpoint.pt` + `metrics.json`.
- Training runs on CPU for toy data (slowly) and on GPU for real data.

---

## Phase 6 — Evaluation + reporting (DONE)

**Spec:** `docs/modules/06_evaluation.md`

Deliverables:
- `src/scatterrad/evaluation/metrics.py` — classification + regression metrics
- `src/scatterrad/evaluation/report.py` — aggregates per-fold metrics.json
  into a summary
- `src/scatterrad/evaluation/plots.py` — optional ROC, calibration, bland-altman
- Wire `cmd_report` in `cli.py`

**Exit criteria:**
- `scatterrad report <dataset> <task>` prints a summary table and writes a
  markdown report.

---

## Phase 7 — Predict (inference on new cases) (DONE)

**Spec:** `docs/modules/07_cli_wiring.md`

Deliverables:
- `src/scatterrad/models/radiomics/predictor.py`
- `src/scatterrad/models/scatter/predictor.py`
- Wire `cmd_predict` in `cli.py`

**Exit criteria:**
- `scatterrad predict <dataset> <task> <input>` works for both tracks.

---

## Phase 8 — Hardening

Across the whole codebase:
- Error messages actionable (tell the user *what to do*)
- Logging at appropriate levels
- Docstrings complete
- README updated with a worked example
- Integration test using a 5-case synthetic dataset end-to-end

---

## Done-criteria for v1

- All 8 phases complete.
- Full test suite passes: `pytest tests/`.
- One worked example runs end-to-end (synthetic dataset committed in
  `examples/`).
- V2 roadmap consulted — nothing implemented from it.

---

## Anti-goals (do not do in v1)

Items from `V2_ROADMAP.md`. If you think one is essential, stop and ask the
user. Otherwise, ignore:

- Multi-modal input
- ComBat harmonization
- Stability filtering
- Self-supervised / contrastive pretraining
- 2D scattering fallback
- Multi-task learning
- Test-time augmentation
- Active learning
- Ensembling beyond CV mean

---

## Time budget (rough)

| Phase | Estimate    |
| ----- | ----------- |
| 0     | 1 session   |
| 1     | 1–2 sessions |
| 2     | 2–3 sessions |
| 3     | 2–3 sessions |
| 4     | 2 sessions   |
| 5     | 3–4 sessions |
| 6     | 1 session    |
| 7     | 1 session    |
| 8     | 1–2 sessions |

If a phase is blowing past estimate, pause and re-spec rather than pushing
through.
