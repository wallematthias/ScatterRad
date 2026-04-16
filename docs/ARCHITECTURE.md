# ARCHITECTURE.md — One-page repo map

## Purpose

ScatterRad is a texture-classification framework for nnunet-style medical
imaging datasets. It wraps a segmentation (already produced by nnunet or
similar) and trains models to predict per-label or per-case outcomes.

Two tracks:
- **Radiomics**: PyRadiomics features → classical ML classifier/regressor.
- **Scatter**: 3D harmonic scattering (fixed) → shallow 3D CNN → attention
  pool across labels → head.

## Repo layout

```
scatterrad/
├── pyproject.toml
├── README.md
├── docs/                       # All specs and decisions. READ FIRST.
│   ├── AGENTS.md
│   ├── ARCHITECTURE.md
│   ├── SCHEMAS.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── V2_ROADMAP.md
│   ├── modules/
│   └── decisions/
├── src/scatterrad/
│   ├── __init__.py
│   ├── cli.py                  # argparse dispatch
│   ├── paths.py                # env-var driven path resolution (DONE)
│   ├── config/                 # DONE — dataset.json, targets.json, task.json parsers
│   ├── preprocessing/          # planner, resample, normalize, crop
│   ├── models/
│   │   ├── radiomics/          # pyradiomics extractor + classical trainer
│   │   └── scatter/            # kymatio frontend + backend + attention
│   ├── data/                   # Dataset, Sampler for training loops
│   ├── evaluation/             # metrics, CV aggregation, reporting
│   └── utils/
└── tests/                      # pytest, one test file per module
```

## Dependency graph (one-way only)

```
  cli.py
    │
    ├──> config/     (stable, stateless)
    ├──> paths.py    (stable, stateless)
    │
    ├──> preprocessing/  ──> config, paths, utils
    │
    ├──> data/           ──> config, paths, utils
    │
    ├──> models/radiomics/ ──> config, data, evaluation
    └──> models/scatter/   ──> config, data, evaluation

  evaluation/          ──> config only
  utils/               ──> (no internal deps)
```

**No cycles. No backwards imports.** If you need a helper used by two sibling
modules, put it in `utils/`.

## Data flow

### Preprocessing (runs once per dataset)

```
Raw NIFTI + labels + targets
         │
         ▼
   [planner] ─── derives spacing, crop_size, normalization stats ─────> plans.json
         │
         ▼
   [preprocessor]
         ├── resample to target spacing
         ├── normalize intensities (modality-aware)
         ├── per-label crop + pad to fixed size
         ├── save .npz (image + mask + meta)
         └── extract radiomics features, cache to .json
         │
         ▼
  preprocessed/crops/<basename>_label<K>.npz
  preprocessed/radiomics/<basename>_label<K>.json
  preprocessed/splits.json         (stratified k-fold, patient-level)
```

### Training — radiomics track

```
  radiomics/*.json + targetsTr/*.json + splits.json
         │
         ▼
   build feature matrix (pandas DataFrame, rows = samples)
         │
         ▼
   per outer fold:
       inner CV hyperparam search (Optuna, limited budget)
       fit best model on outer train
       predict on outer val → metrics
         │
         ▼
  results/<task>__radiomics__fold<F>/
         ├── model.joblib
         ├── metrics.json
         └── predictions.csv
```

### Training — scatter track

```
  crops/*.npz + targetsTr/*.json + splits.json
         │
         ▼
   ScatterDataset (loads crop, applies mask mode)
         │
         ▼
   HarmonicScattering3D (fixed, no grad)  ─── optionally cache output to disk
         │
         ▼
   shallow 3D conv stack (learnable)
         │
         ▼
   per-label feature vector
         │
         ├── [per_label target]  ──> head → loss
         └── [per_case target]   ──> masked attention pool → head → loss
         │
         ▼
  results/<task>__scatter__fold<F>/
         ├── checkpoint.pt
         ├── metrics.json
         └── predictions.csv
```

## Three root directories (env-var driven)

Following nnunet conventions:

| Env var                  | Contents                                            |
| ------------------------ | --------------------------------------------------- |
| `SCATTERRAD_RAW`         | User-provided images, labels, target annotations.   |
| `SCATTERRAD_PREPROCESSED`| Planner output, cached crops and features.          |
| `SCATTERRAD_RESULTS`     | Trained models, predictions, metrics, reports.      |

Each contains per-dataset subdirectories. See `paths.py`.

## Key external dependencies

| Package       | Used in                          | Why                              |
| ------------- | -------------------------------- | -------------------------------- |
| SimpleITK     | preprocessing                    | NIFTI I/O, resampling            |
| pyradiomics   | models/radiomics                 | IBSI-compliant feature extraction |
| kymatio       | models/scatter                   | 3D harmonic scattering transform |
| torch         | models/scatter, data             | backend CNN + training loop       |
| lightgbm      | models/radiomics                 | default classifier/regressor      |
| scikit-learn  | models/radiomics, evaluation     | splits, scalers, metrics          |
| optuna        | models/radiomics, models/scatter | hyperparam search                 |
| pandas        | models/radiomics, evaluation     | feature matrix, reports           |

## Invariants

Anything that breaks these is a bug.

1. Preprocessing is deterministic given the same inputs + plans.
2. Splits are patient-level, generated once, reused across tasks.
3. Crops are centered on the label with configurable margin.
4. Crop size is uniform across a dataset (set by the planner).
5. Radiomics features are cached after first extraction.
6. No crop or feature is ever leaked across folds.
7. Every model artifact is reproducible from `task.json` + `plans.json` + seed.
