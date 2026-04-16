# Module 01: Paths & Config — DONE

**Status:** Implemented in Phase 0. This doc is a reference — do not re-read
the source unless debugging.

## Public API

```python
from scatterrad.paths import ScatterRadPaths

from scatterrad.config import (
    DatasetConfig,     load_dataset_config,
    TargetsSchema,     load_targets_schema,
    CaseTargets,       load_case_targets,
    TargetSpec, TargetType, TargetScope,
    TaskConfig,        load_task_config,
    ModelKind,
)
```

## ScatterRadPaths

Env-var driven path resolver. Three env vars:
- `SCATTERRAD_RAW`
- `SCATTERRAD_PREPROCESSED`
- `SCATTERRAD_RESULTS`

Each contains a per-dataset subdirectory.

```python
paths = ScatterRadPaths.from_env("MyDataset")
paths.dataset_json          # Path to dataset.json
paths.targets_json          # Path to targets.json
paths.targets_tr            # Path to targetsTr/
paths.plans_json            # Path to plans.json (preprocessed)
paths.splits_json           # Path to splits.json (preprocessed)
paths.crop_path("case01", 20)         # .../crops/case01_label020.npz
paths.radiomics_path("case01", 20)    # .../radiomics/case01_label020.json
paths.result_dir("task", "scatter", 0)  # .../task__scatter__fold0/
paths.ensure_preprocessed()
paths.ensure_results()
```

For inference-only workflows, pass `require_raw=False`.

## Config loaders

### `load_dataset_config(path) -> DatasetConfig`

Parses `dataset.json`. Returns immutable dataclass:

```python
cfg.name          # str
cfg.modality      # "CT" or "MR"
cfg.labels        # dict[int, str] — foreground only
cfg.background_id # int (typically 0)
cfg.label_ids     # sorted list[int]
cfg.label_name(20)  # "L1"
```

### `load_targets_schema(path, known_label_ids=None) -> TargetsSchema`

Parses `targets.json`. Cross-validates against dataset labels if provided.

```python
schema["fracture"]              # TargetSpec
schema.names()                  # list of target names
"fracture" in schema            # bool
```

`TargetSpec` fields: `name`, `type` (enum), `scope` (enum), `num_classes`,
`applicable_labels`.

### `load_case_targets(path, schema, basename=None) -> CaseTargets`

Parses one `targetsTr/<basename>.json`.

```python
ct.get_per_case("age")           # float, NaN if missing
ct.get_per_label("fracture", 20) # float, NaN if missing
ct.has_any_for(schema["fracture"]) # bool
```

Missing values are always NaN, never exceptions. Unknown targets and labels
outside `applicable_labels` are silently dropped.

### `load_task_config(path, schema=None) -> TaskConfig`

Parses `task.json`. Schema optional (for validation).

```python
task.name, task.target, task.model  # model is ModelKind enum
task.cv.folds, task.cv.seed
task.labels                         # tuple[int, ...] or None
task.model_config                   # dict[str, Any], passed to model
task.resolved_labels(schema)        # apply precedence rules
```

## When to read the source

- Never, for normal usage.
- For debugging a schema error: read the specific parser function.
- For adding a new config field: read the parser and the test file together.

## When to add a new config

If a future phase needs a new JSON schema (e.g., `eval_config.json`):

1. Add it to `SCHEMAS.md`.
2. Write a parser in `src/scatterrad/config/<name>.py` following the pattern.
3. Re-export from `config/__init__.py`.
4. Write tests next to `test_task_config.py`.

Error classes: `ScatterRadPathError`, `DatasetConfigError`,
`TargetsConfigError`, `TaskConfigError`. Add a new one for each new parser.
