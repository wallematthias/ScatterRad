from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import sys

from scatterrad.config import (
    CVConfig,
    ModelKind,
    PlansConfig,
    TaskConfig,
    load_dataset_config,
    load_targets_schema,
    load_task_config,
)
from scatterrad.paths import ScatterRadPathError, ScatterRadPaths


def _configure_logging(verbosity: int) -> None:
    level = logging.INFO
    if verbosity >= 1:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _paths(dataset: str, require_raw: bool = True) -> ScatterRadPaths:
    return ScatterRadPaths.from_env(dataset, require_raw=require_raw)


def _resolve_dataset_name(dataset_arg: str, require_raw: bool) -> str:
    if dataset_arg.lower().startswith("dataset"):
        return dataset_arg
    if not dataset_arg.isdigit():
        return dataset_arg

    dataset_id = int(dataset_arg)
    prefix = f"Dataset{dataset_id:03d}_"
    candidates: set[str] = set()

    env_names = ["SCATTERRAD_PREPROCESSED", "SCATTERRAD_RESULTS"]
    if require_raw:
        env_names = ["SCATTERRAD_RAW"] + env_names
    for env_name in env_names:
        root = os.environ.get(env_name)
        if not root:
            continue
        root_path = Path(root)
        if not root_path.exists():
            continue
        for item in root_path.glob(f"{prefix}*"):
            if item.is_dir():
                candidates.add(item.name)
    if not candidates:
        return dataset_arg
    if len(candidates) > 1:
        names = ", ".join(sorted(candidates))
        raise ValueError(f"Dataset id {dataset_arg} is ambiguous. Candidates: {names}")
    return next(iter(candidates))


def _get_dataset_arg(args: argparse.Namespace, require_raw: bool) -> str:
    dataset_arg = args.dataset_kw or args.dataset
    if not dataset_arg:
        raise ValueError("Please provide a dataset via --dataset (or positional dataset).")
    return _resolve_dataset_name(dataset_arg, require_raw=require_raw)


def _discover_model_kind(paths: ScatterRadPaths, task_name: str) -> str:
    # New layout: results/<dataset>/<task>/<trainer>__<planner>/fold*/metrics.json
    kinds = set()
    task_root = paths.results_dataset_dir / task_name
    if task_root.exists():
        for metrics_file in sorted(task_root.glob("*/*/metrics.json")):
            try:
                payload = json.loads(metrics_file.read_text())
            except json.JSONDecodeError:
                continue
            model = payload.get("model")
            if isinstance(model, str):
                kinds.add(model)

    if kinds:
        if len(kinds) > 1:
            raise ValueError(f"Task {task_name} has multiple model kinds. Please specify --model-kind")
        return next(iter(kinds))

    # Backward-compatible legacy layout: <task>__<model>__fold<k>
    pattern = re.compile(rf"^{re.escape(task_name)}__(radiomics|scatter)__fold\d+$")
    kinds = set()
    if not paths.results_dataset_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {paths.results_dataset_dir}")
    for item in paths.results_dataset_dir.iterdir():
        m = pattern.match(item.name)
        if m:
            kinds.add(m.group(1))
    if not kinds:
        raise FileNotFoundError(f"No result folds found for task {task_name}")
    if len(kinds) > 1:
        raise ValueError(f"Task {task_name} has multiple model kinds. Please specify --model-kind")
    return next(iter(kinds))


def cmd_validate(args: argparse.Namespace) -> int:
    dataset_name = _get_dataset_arg(args, require_raw=True)
    paths = _paths(dataset_name)
    load_dataset_config(paths.training_dataset_json)
    schema = load_targets_schema(paths.training_targets_json)
    for target_file in sorted(paths.training_targets_tr.glob("*.json")):
        from scatterrad.config import load_case_targets

        load_case_targets(target_file, schema=schema, basename=target_file.stem)
    logging.getLogger(__name__).info("Validation passed for dataset %s", dataset_name)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from scatterrad.preprocessing.planner import plan

    dataset_name = _get_dataset_arg(args, require_raw=True)
    paths = _paths(dataset_name)
    plans = plan(paths, num_workers=args.num_workers, planner=args.planner)
    logging.getLogger(__name__).info("Plans written to %s", paths.plans_json)
    logging.getLogger(__name__).info(
        "Summary: spacing=%s crop=%s labels=%d",
        plans.target_spacing_mm,
        plans.crop_size_voxels,
        len(plans.label_coverage),
    )
    if args.planner:
        logging.getLogger(__name__).info("Planner argument provided: %s", args.planner)
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from scatterrad.preprocessing.runner import preprocess

    dataset_name = _get_dataset_arg(args, require_raw=True)
    paths = _paths(dataset_name)
    preprocess(paths, num_workers=args.num_workers)
    return 0


def cmd_preprocess_test(args: argparse.Namespace) -> int:
    from scatterrad.preprocessing.runner import preprocess_test

    dataset_name = _get_dataset_arg(args, require_raw=True)
    paths = _paths(dataset_name)
    preprocess_test(paths, num_workers=args.num_workers)
    return 0


def cmd_generate_holdout(args: argparse.Namespace) -> int:
    from scatterrad.preprocessing.holdout import generate_holdout

    dataset_name = _get_dataset_arg(args, require_raw=True)
    paths = _paths(dataset_name)
    payload = generate_holdout(
        paths=paths,
        holdout_fraction=float(args.fraction),
        seed=int(args.seed),
        move_files=not bool(args.no_move),
    )
    logging.getLogger(__name__).info(
        "Generated holdout: %d train / %d holdout",
        len(payload["train_cases"]),
        len(payload["holdout_cases"]),
    )
    return 0


def cmd_scatter_cache(args: argparse.Namespace) -> int:
    from scatterrad.models.scatter.frontend import ScatterFrontend
    from scatterrad.models.scatter.scatter_cache import precompute_and_cache
    from scatterrad.config import PlansConfig

    dataset_name = _get_dataset_arg(args, require_raw=False)
    paths = _paths(dataset_name, require_raw=False)
    plans = PlansConfig.from_json(paths.plans_json)

    if args.force:
        import shutil
        cache_dir = paths.preprocessed_dataset_dir / "scatter_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logging.getLogger(__name__).info("Cleared existing scatter cache at %s", cache_dir)

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    log_sigmas_mm = tuple(float(s) for s in args.log_sigmas.split(",")) if args.log_sigmas else (1.0, 2.0, 3.0)
    spacing_mm = tuple(float(s) for s in plans.target_spacing_mm)
    frontend = ScatterFrontend(
        crop_size=tuple(plans.crop_size_voxels),
        spacing_mm=spacing_mm,
        wavelet=args.wavelet,
        level=args.level,
        log_sigmas_mm=log_sigmas_mm,
        use_gradient=not args.no_gradient,
        mask_mode="zero",
    )
    logging.getLogger(__name__).info(
        "Filter-bank frontend: wavelet=%s level=%d  log_sigmas=%s  gradient=%s  "
        "out_channels=%d  spacing_mm=%s  device=%s  cache_aug_variants=%d",
        args.wavelet, args.level, log_sigmas_mm, not args.no_gradient,
        frontend.out_channels, spacing_mm, device, int(args.augment_variants),
    )
    precompute_and_cache(
        paths=paths,
        frontend=frontend,
        device=device,
        num_augmented_variants=int(args.augment_variants),
        cache_aug_seed=int(args.augment_seed),
        intensity_scale_delta=float(args.aug_intensity_scale),
        intensity_shift_delta=float(args.aug_intensity_shift),
        noise_std=float(args.aug_noise_std),
        elastic_alpha=float(args.aug_elastic_alpha),
        elastic_sigma=float(args.aug_elastic_sigma),
    )
    logging.getLogger(__name__).info("Scatter cache written to %s", paths.preprocessed_dataset_dir / "scatter_cache")
    return 0


def cmd_radiomics_perturb(args: argparse.Namespace) -> int:
    from scatterrad.config import load_dataset_config
    from scatterrad.models.radiomics.reproducibility import compute_reproducibility_icc

    dataset_name = _get_dataset_arg(args, require_raw=False)
    paths = _paths(dataset_name, require_raw=False)
    dataset = load_dataset_config(paths.training_dataset_json)
    out = compute_reproducibility_icc(
        paths=paths,
        modality=dataset.modality,
        n_perturb=int(args.n_perturb),
        max_cases=int(args.max_cases),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
    )
    logging.getLogger(__name__).info("Radiomics reproducibility file: %s", out)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    dataset_name = _get_dataset_arg(args, require_raw=False)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        logging.getLogger(__name__).info("Using CUDA_VISIBLE_DEVICES=%s", args.gpu)
    paths = _paths(dataset_name, require_raw=False)
    dataset = load_dataset_config(paths.training_dataset_json)
    schema = load_targets_schema(
        paths.training_targets_json, known_label_ids=set(dataset.label_ids)
    )
    if args.task is not None:
        task = load_task_config(args.task, schema=schema)
    else:
        target = args.target or sorted(schema.names())[0]
        if target not in schema:
            raise ValueError(f"Unknown target '{target}'. Available: {', '.join(schema.names())}")
        model_kind = ModelKind(args.model)
        task = TaskConfig(
            name=f"{target}_{model_kind.value}_auto",
            target=target,
            model=model_kind,
            cv=CVConfig(folds=int(args.folds), seed=int(args.seed)),
            labels=None,
            model_config={},
        )
        logging.getLogger(__name__).info("Auto-generated task name: %s", task.name)

    if task.model is ModelKind.SCATTER:
        if args.cache_aug_variants is not None:
            task.model_config["cache_aug_variants"] = int(args.cache_aug_variants)
        if bool(args.debug):
            task.model_config["debug"] = True
        if args.debug_every is not None:
            task.model_config["debug_save_every_n_epochs"] = int(args.debug_every)
        if args.debug_cases is not None:
            task.model_config["debug_num_cases"] = int(args.debug_cases)
    plans = PlansConfig.from_json(paths.plans_json)

    if task.model is ModelKind.RADIOMICS:
        from scatterrad.models.radiomics.trainer import train

        if args.resume_from is not None:
            raise ValueError("--resume-from is only supported for scatter model training")
        train(paths, task, dataset, schema, plans, fold=args.fold, continue_existing=args.cont)
    else:
        from scatterrad.models.scatter.trainer import train

        train(
            paths,
            task,
            dataset,
            schema,
            plans,
            fold=args.fold,
            continue_existing=args.cont,
            resume_from=args.resume_from,
        )
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    from scatterrad.evaluation.aggregate import aggregate_folds

    dataset_name = _get_dataset_arg(args, require_raw=False)
    paths = _paths(dataset_name, require_raw=False)
    model_kind = args.model or _discover_model_kind(paths, args.task_name)
    fold_dirs = paths.result_fold_dirs(args.task_name, model_kind)
    if not fold_dirs:
        logging.getLogger(__name__).error(
            "No fold results found for task '%s' model '%s'", args.task_name, model_kind
        )
        return 2

    agg = aggregate_folds(fold_dirs)
    summary = agg["metrics_summary"]

    print(f"\nCV summary — task: {args.task_name}  model: {model_kind}  folds: {agg['n_folds']}\n")
    col_w = max(len(k) for k in summary) + 2
    print(f"{'metric':<{col_w}}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    print("-" * (col_w + 38))
    for metric, stats in sorted(summary.items()):
        if not isinstance(stats, dict):
            continue
        print(
            f"{metric:<{col_w}}  {stats['mean']:>8.4f}  {stats['std']:>8.4f}"
            f"  {stats['min']:>8.4f}  {stats['max']:>8.4f}"
        )
    print()
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    dataset_name = _get_dataset_arg(args, require_raw=False)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        logging.getLogger(__name__).info("Using CUDA_VISIBLE_DEVICES=%s", args.gpu)
    paths = _paths(dataset_name, require_raw=False)
    dataset = load_dataset_config(paths.training_dataset_json)
    schema = load_targets_schema(
        paths.training_targets_json, known_label_ids=set(dataset.label_ids)
    )
    if args.task is not None:
        task = load_task_config(args.task, schema=schema)
    else:
        target = args.target or sorted(schema.names())[0]
        if target not in schema:
            raise ValueError(f"Unknown target '{target}'. Available: {', '.join(schema.names())}")
        model_kind = ModelKind(args.model)
        task = TaskConfig(
            name=f"{target}_{model_kind.value}_auto",
            target=target,
            model=model_kind,
            cv=CVConfig(folds=5, seed=42),
            labels=None,
            model_config={},
        )
        logging.getLogger(__name__).info("Auto-generated task name: %s", task.name)
    plans = PlansConfig.from_json(paths.plans_json)

    if task.model is ModelKind.RADIOMICS:
        from scatterrad.models.radiomics.tester import test
        test(paths, task, dataset, schema, plans)
    else:
        raise NotImplementedError("test command is currently only supported for radiomics model")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from scatterrad.evaluation.report import render_report

    dataset_name = _get_dataset_arg(args, require_raw=False)
    paths = _paths(dataset_name, require_raw=False)
    model_kind = args.model_kind or _discover_model_kind(paths, args.task_name)
    md = render_report(paths, args.task_name, model_kind)
    print(md, end="")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    dataset_name = _get_dataset_arg(args, require_raw=False)
    paths = _paths(dataset_name, require_raw=False)
    model_kind = args.model_kind or _discover_model_kind(paths, args.task_name)

    inputs: list[Path]
    if args.input.is_dir():
        inputs = sorted(p for p in args.input.iterdir() if p.is_file())
    else:
        inputs = [args.input]

    if model_kind == "radiomics":
        from scatterrad.models.radiomics.predictor import predict
    else:
        from scatterrad.models.scatter.predictor import predict

    df = predict(paths, args.task_name, model_kind, inputs, fold=args.fold)
    if args.output:
        df.to_csv(args.output, index=False)
    else:
        print(df.to_csv(index=False), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scatterrad")
    parser.add_argument("-v", "--verbose", action="count", default=0)

    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("dataset", nargs="?")
    p_validate.add_argument("--dataset", dest="dataset_kw")
    p_validate.set_defaults(func=cmd_validate)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("dataset", nargs="?")
    p_plan.add_argument("--dataset", dest="dataset_kw")
    p_plan.add_argument("--planner", default="all")
    p_plan.add_argument("--num-workers", type=int, default=0)
    p_plan.set_defaults(func=cmd_plan)

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("dataset", nargs="?")
    p_pre.add_argument("--dataset", dest="dataset_kw")
    p_pre.add_argument("--num-workers", type=int, default=0)
    p_pre.set_defaults(func=cmd_preprocess)

    p_pre_test = sub.add_parser("preprocess-test", help="Preprocess test-set cases (imagesTs/labelsTs)")
    p_pre_test.add_argument("dataset", nargs="?")
    p_pre_test.add_argument("--dataset", dest="dataset_kw")
    p_pre_test.add_argument("--num-workers", type=int, default=0)
    p_pre_test.set_defaults(func=cmd_preprocess_test)

    p_holdout = sub.add_parser("generate-holdout")
    p_holdout.add_argument("dataset", nargs="?")
    p_holdout.add_argument("--dataset", dest="dataset_kw")
    p_holdout.add_argument("--fraction", type=float, default=0.2)
    p_holdout.add_argument("--seed", type=int, default=42)
    p_holdout.add_argument(
        "--no-move",
        action="store_true",
        help="Only generate test_split_manifest.json; do not move files from *Tr to *Ts.",
    )
    p_holdout.set_defaults(func=cmd_generate_holdout)

    p_scatter_cache = sub.add_parser("scatter-cache", help="Precompute (or recompute) scatter transform cache")
    p_scatter_cache.add_argument("dataset", nargs="?")
    p_scatter_cache.add_argument("--dataset", dest="dataset_kw")
    p_scatter_cache.add_argument("--wavelet", default="coif1", help="PyWavelets wavelet name (default: coif1)")
    p_scatter_cache.add_argument("--level", type=int, default=1, help="Decomposition levels (default: 1 → 8 channels)")
    p_scatter_cache.add_argument("--log-sigmas", dest="log_sigmas", default="1.0,2.0,3.0",
                                  help="Comma-separated LoG sigma values in mm (default: 1.0,2.0,3.0)")
    p_scatter_cache.add_argument("--no-gradient", action="store_true", help="Disable gradient magnitude channel")
    p_scatter_cache.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    p_scatter_cache.add_argument("--force", action="store_true", help="Delete existing cache and recompute")
    p_scatter_cache.add_argument(
        "--augment-variants",
        type=int,
        default=0,
        help="Number of extra augmented cache variants to generate per crop (default: 0).",
    )
    p_scatter_cache.add_argument(
        "--augment-seed",
        type=int,
        default=42,
        help="Seed for deterministic cache augmentation variants (default: 42).",
    )
    p_scatter_cache.add_argument(
        "--aug-intensity-scale",
        type=float,
        default=0.1,
        help="Max multiplicative intensity jitter delta around 1.0 for augmented cache variants.",
    )
    p_scatter_cache.add_argument(
        "--aug-intensity-shift",
        type=float,
        default=0.1,
        help="Max additive intensity jitter magnitude for augmented cache variants.",
    )
    p_scatter_cache.add_argument(
        "--aug-noise-std",
        type=float,
        default=0.05,
        help="Upper bound of Gaussian noise sigma for augmented cache variants.",
    )
    p_scatter_cache.add_argument(
        "--aug-elastic-alpha",
        type=float,
        default=1.0,
        help="Elastic displacement magnitude in voxels for augmented cache variants.",
    )
    p_scatter_cache.add_argument(
        "--aug-elastic-sigma",
        type=float,
        default=6.0,
        help="Elastic smoothing sigma in voxels for augmented cache variants.",
    )
    p_scatter_cache.set_defaults(func=cmd_scatter_cache)

    p_perturb = sub.add_parser("radiomics-perturb")
    p_perturb.add_argument("dataset", nargs="?")
    p_perturb.add_argument("--dataset", dest="dataset_kw")
    p_perturb.add_argument("--n-perturb", type=int, default=8)
    p_perturb.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit number of crops for faster ICC estimation (0 = all).",
    )
    p_perturb.add_argument("--seed", type=int, default=42)
    p_perturb.add_argument("--num-workers", type=int, default=0)
    p_perturb.set_defaults(func=cmd_radiomics_perturb)

    p_train = sub.add_parser("train")
    p_train.add_argument("dataset", nargs="?")
    p_train.add_argument("--dataset", dest="dataset_kw")
    p_train.add_argument("--task", type=Path, default=None)
    p_train.add_argument("--model", choices=["radiomics", "scatter"], default="radiomics")
    p_train.add_argument("--target", default=None)
    p_train.add_argument("--folds", type=int, default=5)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--fold", type=int, default=None)
    p_train.add_argument("-c", "--continue", dest="cont", action="store_true")
    p_train.add_argument("--resume-from", type=Path, default=None)
    p_train.add_argument("--gpu", type=int, default=None)
    p_train.add_argument(
        "--cache-aug-variants",
        type=int,
        default=None,
        help="Scatter model: number of extra cached augmentation variants to sample during training.",
    )
    p_train.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Scatter model: save debug input-feature and attention panels to fold/debug/.",
    )
    p_train.add_argument(
        "--debug-every",
        type=int,
        default=None,
        help="Scatter model: save debug panels every N epochs.",
    )
    p_train.add_argument(
        "--debug-cases",
        type=int,
        default=None,
        help="Scatter model: number of validation cases to render per debug save.",
    )
    p_train.set_defaults(func=cmd_train)

    p_summary = sub.add_parser("summary", help="Print mean±std CV metrics across folds")
    p_summary.add_argument("dataset", nargs="?")
    p_summary.add_argument("--dataset", dest="dataset_kw")
    p_summary.add_argument("task_name")
    p_summary.add_argument("--model", choices=["radiomics", "scatter"], default=None)
    p_summary.set_defaults(func=cmd_summary)

    p_test = sub.add_parser("test", help="Ensemble test-set evaluation using all fold models")
    p_test.add_argument("dataset", nargs="?")
    p_test.add_argument("--dataset", dest="dataset_kw")
    p_test.add_argument("--task", type=Path, default=None)
    p_test.add_argument("--model", choices=["radiomics", "scatter"], default="radiomics")
    p_test.add_argument("--target", default=None)
    p_test.add_argument("--gpu", type=int, default=None)
    p_test.set_defaults(func=cmd_test)

    p_report = sub.add_parser("report")
    p_report.add_argument("dataset", nargs="?")
    p_report.add_argument("--dataset", dest="dataset_kw")
    p_report.add_argument("task_name")
    p_report.add_argument("--model-kind", choices=["radiomics", "scatter"], default=None)
    p_report.set_defaults(func=cmd_report)

    p_predict = sub.add_parser("predict")
    p_predict.add_argument("dataset", nargs="?")
    p_predict.add_argument("--dataset", dest="dataset_kw")
    p_predict.add_argument("task_name")
    p_predict.add_argument("input", type=Path)
    p_predict.add_argument("--fold", type=int, default=None)
    p_predict.add_argument("--model-kind", choices=["radiomics", "scatter"], default=None)
    p_predict.add_argument("--output", type=Path, default=None)
    p_predict.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        return int(args.func(args))
    except (ScatterRadPathError, FileNotFoundError, ValueError) as exc:
        logging.getLogger(__name__).error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
