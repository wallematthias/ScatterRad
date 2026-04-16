# Module 07: CLI Wiring

**Status:** Partially implemented in Phase 0 (skeleton only). Finished across
Phases 1–7 as each feature lands.

**Depends on:** everything.

---

## Overview

Connects each module to the CLI. `cli.py` already has argparse scaffolding
and stub handlers. Each phase replaces one stub with a real implementation.

---

## Subcommand wiring checklist

### `scatterrad plan <dataset>` (Phase 1)

```python
def cmd_plan(args):
    paths = _paths(args.dataset)
    from scatterrad.preprocessing.planner import plan
    plans = plan(paths)
    log.info("Plans written to %s", paths.plans_json)
    log.info("Summary: spacing=%s, crop_size=%s, labels=%d",
             plans.target_spacing_mm, plans.crop_size_voxels,
             len(plans.label_coverage))
    return 0
```

### `scatterrad preprocess <dataset>` (Phase 2)

```python
def cmd_preprocess(args):
    paths = _paths(args.dataset)
    from scatterrad.preprocessing.runner import preprocess
    preprocess(paths, num_workers=args.num_workers)
    return 0
```

Add optional args in build_parser:
- `--stratify <target_name>` — override auto-stratification for splits.

### `scatterrad train <dataset> <task.json>` (Phases 3, 5)

```python
def cmd_train(args):
    paths = _paths(args.dataset)
    ds = load_dataset_config(paths.dataset_json)
    schema = load_targets_schema(paths.targets_json, known_label_ids=set(ds.label_ids))
    task = load_task_config(args.task, schema=schema)
    plans = PlansConfig.from_json(paths.plans_json)

    if task.model is ModelKind.RADIOMICS:
        from scatterrad.models.radiomics.trainer import train
    else:
        from scatterrad.models.scatter.trainer import train

    train(paths, task, ds, schema, plans, fold=args.fold)
    return 0
```

### `scatterrad predict` (Phase 7)

```python
def cmd_predict(args):
    paths = _paths(args.dataset, require_raw=False)
    # Discover model kind from results directory name
    # Dispatch to appropriate predictor
    ...
```

### `scatterrad report <dataset> <task>` (Phase 6)

```python
def cmd_report(args):
    paths = _paths(args.dataset, require_raw=False)
    from scatterrad.evaluation.report import render_report
    # Discover model_kind by looking in results dir
    model_kind = _discover_model_kind(paths, args.task_name)
    md = render_report(paths, args.task_name, model_kind)
    print(md)
    return 0
```

### `scatterrad validate` (Phase 0, done)

Already implemented.

---

## Argument conventions

- Dataset name is always first positional arg (after subcommand).
- Paths to files use `type=Path`.
- Flags that default to "use-all" → `--fold INT` for single-fold, absent = all.
- `--num-workers` where parallelism applies, default 0 (serial, debuggable).
- `-v` / `-vv` for verbosity already set at top level.

---

## Error handling

The top-level `main()` already wraps dispatch in try/except. Specific
handlers should:
- Return `0` on success.
- Return `2` for user-input problems (missing files, bad configs).
- Let exceptions propagate for unexpected errors — `main()` logs them.

---

## Logging

Per-module loggers: `log = logging.getLogger(__name__)`.

Use `log.info` for progress (visible with `-v`), `log.debug` for deep detail
(visible with `-vv`), `log.warning` for recoverable issues, `log.error` for
problems that prevent completion.

Training loops should print per-epoch progress via `tqdm`, not `log.info` —
tqdm handles the terminal better.

---

## Tests

Keep `test_cli_smoke.py` passing at every phase. As commands get real
implementations, add integration-style tests using a tiny synthetic dataset
fixture.

Don't test heavy ML paths at the CLI level — unit-test the internals, smoke-
test the wiring.

---

## What NOT to do

- Do not add subcommands not in `build_parser()`. If a new feature needs one,
  update this doc first.
- Do not parse args inside submodules. Submodules take typed arguments.
- Do not bypass the config parsers and read JSON directly. Always go through
  `config/`.
