# V2_ROADMAP.md — Deferred features

Everything here is **out of scope for v1**. If a user explicitly asks to
build one of these, consult this doc first for context. Otherwise ignore.

Items are grouped by theme and sized (S/M/L). Each has a "trigger" — the
condition under which it becomes worth building.

---

## Preprocessing

### N4 bias correction for MR (M)

**Why deferred:** adds dependency on ANTs/SimpleITK's N4 routine, slow per
case, default not always better for surface-coil images.

**Trigger:** any MR-on-ScatterRad user reports heterogeneity issues or poor
radiomics reproducibility.

**Sketch:** `preprocessing/n4.py`, optional flag in `plans.json`, applied
before normalize.

### Nyul histogram standardization for MR (M)

**Why deferred:** cross-scanner problem doesn't matter for single-site datasets.

**Trigger:** multi-center MR dataset with obvious scanner bias.

**Sketch:** fit Nyul landmarks during `plan`, apply during `preprocess`.

### Configurable CT clipping windows (S)

**Why deferred:** general `[-1000, 1000]` works for most tasks; per-organ
tuning is fiddly.

**Trigger:** user with specialized organ focus (lung texture, bone density).

**Sketch:** add `intensity_clip` field in `task.json.model_config` that
overrides the planner default.

### ComBat harmonization (L)

**Why deferred:** requires scanner/site metadata in targetsTr; most users
won't have it; adds `neuroHarmonize` or custom implementation dep.

**Trigger:** multi-center validation fails despite preprocessing.

**Sketch:** `preprocessing/combat.py`, reads `scanner_id` from targetsTr,
applies parametric or empirical Bayes ComBat pre-feature-selection.

---

## Input modalities

### Multi-channel input (L)

**Why deferred:** see DEC001. Clean branching requires per-channel
normalization and rewriting parts of the pipeline.

**Trigger:** real use case for MR multi-sequence (T1+T2+FLAIR) or CT+PET.

**Sketch:** extend `dataset.json.modality` to allow multiple entries, per-
channel normalize, concat at input of scatter frontend.

### 2D scattering fallback (M)

**Why deferred:** v1 is 3D-only.

**Trigger:** 2D modality dataset (mammography, dermatology) comes up.

**Sketch:** kymatio.torch.Scattering2D, `ScatterFrontend2D`, separate
pipeline branch.

---

## Feature / model additions (radiomics track)

### Stability filtering (M)

**Why deferred:** requires either test-retest data or ROI perturbation
infrastructure.

**Trigger:** user has test-retest or is willing to tolerate ROI
perturbation overhead.

**Sketch:** `models/radiomics/stability.py` that shakes the ROI (erode/
dilate, small rotations), computes ICC per feature, filters below threshold.

### Contrastive / SimCLR pretraining (L)

**Why deferred:** extra compute cost and complexity that doesn't obviously
help small-data handcrafted radiomics.

**Trigger:** user wants to pretrain a CNN feature extractor on unlabeled
data before fine-tuning.

**Sketch:** `models/contrastive/` track with a third model kind.
Replaces handcrafted radiomics with learned embeddings.

### Deep radiomics encoder (L)

**Why deferred:** explicitly out of scope for v1.

**Trigger:** clear evidence scatter track is overfitting; need learned
features without falling back to large CNNs.

---

## Feature / model additions (scatter track)

### Multi-head / gated attention pooling (S)

**Why deferred:** masked attention is sufficient for v1.

**Trigger:** per-case tasks plateau; label interactions look important.

**Sketch:** swap `MaskedAttentionPool` for a multi-head variant. Config flag.

### Learnable scattering (M)

**Why deferred:** defeats the "fixed front-end, small-data friendly"
purpose.

**Trigger:** large-data regime where scatter is now a bottleneck.

### Deeper backend (S)

**Why deferred:** two-block design is intentional for small-data.

**Trigger:** >5000 cases consistently available.

### Augmentation-compatible scatter cache (M)

**Why deferred:** v1 mutually excludes cache + augmentation (see DEC005).

**Trigger:** user wants both and training is slow.

**Sketch:** pre-compute N augmented versions per crop, store all, sample at
load time.

### Elastic deformation augmentation (M)

**Why deferred:** breaks scattering's deformation invariance guarantees.

**Trigger:** user insists on elastic aug for a scatter model (probably a
mistake; consider educating first).

---

## Training infrastructure

### Multi-task learning (L)

**Why deferred:** one task per task.json is cleaner and matches nnunet.

**Trigger:** user wants to jointly predict fracture + metastasis + age and
share a backbone.

**Sketch:** `task.json` allows `targets: [list]` instead of `target: str`.
Head becomes multi-output.

### Test-time augmentation (S)

**Why deferred:** typically small gains; adds inference complexity.

**Trigger:** user wants the last few points of AUC.

### Ensembling beyond CV mean (S)

**Why deferred:** reporting CV mean is cleaner.

**Trigger:** user wants a single deployable model across all folds.

**Sketch:** average predictions or logits across the N fold models.

### Active learning (L)

**Why deferred:** out of scope for a training framework.

**Trigger:** never, probably — different tool.

### Per-layer LR scheduling, SAM, lookahead, etc. (S)

**Why deferred:** diminishing returns on small-data.

---

## Uncertainty / calibration

### Monte Carlo dropout uncertainty (M)

**Trigger:** clinical deployment context where uncertainty matters.

### Platt scaling / isotonic calibration (S)

**Trigger:** classification outputs look uncalibrated.

---

## Reporting

### Plotted reports (ROC, calibration, Bland-Altman) (S)

**Why deferred:** v1 markdown report is terminal-friendly.

**Trigger:** user wants publication-ready plots.

**Sketch:** `evaluation/plots.py` with matplotlib, saved alongside report.md.

### HTML / PDF report (M)

**Trigger:** user wants to share with non-technical collaborators.

### W&B / MLflow integration (S)

**Trigger:** user in a tracked-experiments environment.

---

## Deployment

### Inference server (L)

**Why deferred:** training framework, not a deployment tool.

### DICOM I/O (M)

**Trigger:** user has DICOM-native pipeline; NIFTI conversion is friction.

---

## Other ideas worth revisiting

- **Radiomics + scatter hybrid** — concatenate features from both tracks into
  the final classifier. Could be a strong ensemble. Sizing: M.
- **Per-label CNN backbone weights** — separate backbone per label ID
  instead of shared. Only makes sense for strongly different anatomies.
  Sizing: M.
- **Cross-dataset transfer** — pretrain backbone on dataset A, fine-tune on
  dataset B. Useful for rare-disease small-N. Sizing: L.
