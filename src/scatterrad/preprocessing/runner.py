from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from scatterrad.config import TargetType, PlansConfig, load_case_targets, load_targets_schema
from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.crop import bbox_from_label, crop_around_bbox, write_crop
from scatterrad.preprocessing.normalize import normalize
from scatterrad.preprocessing.resample import resample_to_spacing
from scatterrad.preprocessing.splits import generate_splits
from scatterrad.utils import resolve_num_workers

log = logging.getLogger(__name__)


def _case_basenames(images_tr: Path) -> list[str]:
    return sorted(p.name.replace("_0000.nii.gz", "") for p in images_tr.glob("*_0000.nii.gz"))


def _margin_voxels(
    margin_mm: float, spacing_zyx: tuple[float, float, float]
) -> tuple[int, int, int]:
    return tuple(int(np.ceil(margin_mm / s)) for s in spacing_zyx)


def _extract_scalar_value(raw: dict[str, Any], key: str) -> Any:
    nested = raw.get("targets", {})
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return raw.get(key)


def _collect_class_value_shift(
    raw_target_payloads: dict[str, dict[str, Any]], target_name: str, num_classes: int
) -> int:
    values: list[int] = []
    for payload in raw_target_payloads.values():
        raw_value = _extract_scalar_value(payload, target_name)
        if raw_value is None:
            continue
        try:
            values.append(int(raw_value))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0
    if min(values) >= 1 and max(values) == num_classes:
        return 1
    return 0


def _normalize_class_value(value: Any, num_classes: int, shift: int) -> Any:
    if value is None:
        return None
    try:
        val = int(value)
    except (TypeError, ValueError):
        return value
    if shift == 1:
        val = val - 1
    if val < 0 or val >= num_classes:
        return value
    return val


def _preprocess_case(
    image_path: str,
    label_path: str,
    basename: str,
    target_spacing_mm: tuple[float, float, float],
    crop_margin_mm: float,
    crop_size_voxels: tuple[int, int, int],
    modality: str,
    intensity_clip: tuple[float, float] | None,
    intensity_mean: float | None,
    intensity_std: float | None,
    label_ids: tuple[int, ...],
    crop_paths: tuple[str, ...],
) -> tuple[bool, bool]:
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    image_rs = resample_to_spacing(image, target_spacing_mm, is_label=False)
    label_rs = resample_to_spacing(label, target_spacing_mm, is_label=True)

    x = sitk.GetArrayFromImage(image_rs).astype(np.float32)
    y = sitk.GetArrayFromImage(label_rs).astype(np.int16)
    if x.shape != y.shape:
        raise ValueError(
            f"Resampled shape mismatch for {basename}: image={x.shape}, label={y.shape}"
        )

    plans_case = PlansConfig(
        version=1,
        dataset_name="",
        modality=modality,
        target_spacing_mm=target_spacing_mm,
        crop_size_voxels=crop_size_voxels,
        crop_margin_mm=crop_margin_mm,
        intensity_clip=intensity_clip,
        intensity_mean=intensity_mean,
        intensity_std=intensity_std,
        orientation="RAS",
        label_coverage={},
        bbox_percentiles={},
    )
    fg = (y > 0).astype(np.uint8)
    x_norm = normalize(x, fg, plans_case)

    spacing_zyx = tuple(float(v) for v in reversed(image_rs.GetSpacing()))
    # Tight crops only: no additional margin around bbox.
    _ = crop_margin_mm
    margins = (0, 0, 0)
    found_foreground = False
    wrote_any = False
    for label_id, crop_path_str in zip(label_ids, crop_paths):
        bbox = bbox_from_label(y, label_id)
        if bbox is None:
            continue
        found_foreground = True
        crop_file = Path(crop_path_str)
        if crop_file.exists():
            continue
        mask = (y == label_id).astype(np.uint8)
        crop_img, crop_msk = crop_around_bbox(
            x_norm,
            mask,
            bbox=bbox,
            margin_voxels=margins,
        )
        meta = {
            "basename": basename,
            "label_id": int(label_id),
            "original_spacing_mm": spacing_zyx,
            "bbox_voxels": [(s.start, s.stop) for s in bbox],
            "crop_shape_zyx": tuple(int(v) for v in crop_img.shape),
        }
        write_crop(crop_file, crop_img, crop_msk, meta)
        wrote_any = True
    _ = crop_size_voxels
    return found_foreground, wrote_any


def _planned_cache_modes(planner_name: str | None) -> tuple[bool, bool]:
    if not planner_name:
        return False, False
    name = planner_name.strip().lower()
    if not name:
        return False, False
    if "all" in name or "both" in name:
        return True, True
    do_radiomics = "radiomics" in name
    do_scatter = ("scatter" in name) or ("wave" in name)
    return do_radiomics, do_scatter


def _run_parallel_cases(
    fn,
    args_list: list[tuple],
    workers: int,
    desc: str,
):
    if workers <= 1:
        return [fn(*args) for args in tqdm(args_list, total=len(args_list), desc=desc, leave=False)]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, *args) for args in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
            results.append(fut.result())
    return results


def preprocess(paths: ScatterRadPaths, num_workers: int = 0) -> None:
    """Run preprocessing pipeline and write crops + splits."""
    if not paths.plans_json.exists():
        raise FileNotFoundError(
            f"{paths.plans_json} missing. Run `scatterrad plan {paths.dataset_name}` first."
        )

    plans = PlansConfig.from_json(paths.plans_json)
    schema = load_targets_schema(paths.targets_json)
    basenames = _case_basenames(paths.images_tr)
    paths.ensure_preprocessed()
    # No backward compatibility mode: regenerate preprocessing artifacts fresh.
    for stale in paths.crops_dir.glob("*.npz"):
        stale.unlink()
    for stale in paths.radiomics_dir.glob("*.json"):
        stale.unlink()
    scatter_cache_dir = paths.preprocessed_dataset_dir / "scatter_cache"
    if scatter_cache_dir.exists():
        for stale in scatter_cache_dir.glob("*.npy"):
            stale.unlink()

    raw_target_payloads: dict[str, dict[str, Any]] = {}
    for basename in basenames:
        raw_path = paths.targets_tr / f"{basename}.json"
        if raw_path.exists():
            raw_target_payloads[basename] = json.loads(raw_path.read_text())

    class_shifts: dict[str, int] = {}
    for target_name, spec in schema.items():
        if spec.type is TargetType.CLASSIFICATION:
            class_shifts[target_name] = _collect_class_value_shift(
                raw_target_payloads=raw_target_payloads,
                target_name=target_name,
                num_classes=int(spec.num_classes or 2),
            )

    # Persist schema metadata in preprocessed so training can run without raw data.
    paths.preprocessed_dataset_json.write_text(paths.dataset_json.read_text())
    paths.preprocessed_targets_json.write_text(paths.targets_json.read_text())

    # Persist canonicalized per-case targets in preprocessed.
    for basename, payload in raw_target_payloads.items():
        out = dict(payload)
        for target_name, spec in schema.items():
            raw_value = _extract_scalar_value(payload, target_name)
            if spec.type is TargetType.CLASSIFICATION:
                raw_value = _normalize_class_value(
                    value=raw_value,
                    num_classes=int(spec.num_classes or 2),
                    shift=class_shifts.get(target_name, 0),
                )
            elif raw_value is not None:
                try:
                    raw_value = float(raw_value)
                except (TypeError, ValueError):
                    raw_value = None
            out[target_name] = raw_value
        (paths.preprocessed_targets_tr / f"{basename}.json").write_text(
            json.dumps(out, indent=2) + "\n"
        )

    all_targets = {
        b: load_case_targets(paths.preprocessed_targets_tr / f"{b}.json", schema=schema, basename=b)
        for b in basenames
        if (paths.preprocessed_targets_tr / f"{b}.json").exists()
    }

    label_ids = tuple(int(v) for v in plans.label_coverage)
    case_args = [
        (
            str(paths.images_tr / f"{basename}_0000.nii.gz"),
            str(paths.labels_tr / f"{basename}.nii.gz"),
            basename,
            tuple(plans.target_spacing_mm),
            float(plans.crop_margin_mm),
            tuple(plans.crop_size_voxels),
            plans.modality,
            plans.intensity_clip,
            plans.intensity_mean,
            plans.intensity_std,
            label_ids,
            tuple(str(paths.crop_path(basename, label_id)) for label_id in label_ids),
        )
        for basename in basenames
    ]
    if not case_args:
        raise ValueError(f"No training cases in {paths.images_tr}; expected *_0000.nii.gz files")
    workers = resolve_num_workers(num_workers, max_tasks=len(case_args))
    case_flags = _run_parallel_cases(_preprocess_case, case_args, workers=workers, desc="preprocess")

    for basename, (found_foreground, _wrote_any) in zip(basenames, case_flags):
        if not found_foreground:
            log.warning("No foreground labels for case %s", basename)

    first_cls = None
    for name, spec in schema.items():
        if spec.type.value == "classification":
            first_cls = name
            break
    folds = generate_splits(
        basenames=basenames,
        targets=all_targets,
        schema=schema,
        n_folds=5,
        seed=42,
        stratification_target=first_cls,
    )

    payload = {
        "seed": 42,
        "n_folds": 5,
        "strategy": "patient_stratified_kfold" if first_cls else "kfold",
        "stratification_key": first_cls,
        "folds": folds,
    }
    paths.splits_json.write_text(json.dumps(payload, indent=2) + "\n")
    paths.ensure_results()
    paths.results_splits_json.write_text(json.dumps(payload, indent=2) + "\n")

    do_radiomics, do_scatter = _planned_cache_modes(plans.planner)
    if do_radiomics:
        from scatterrad.models.radiomics.analysis import compute_intercorrelation
        from scatterrad.models.radiomics.extractor import extract_all
        from scatterrad.models.radiomics.reproducibility import compute_reproducibility_icc

        log.info("Plan requests radiomics preprocessing; extracting radiomics cache.")
        extract_all(paths=paths, modality=plans.modality, num_workers=workers, force=False)
        # End-to-end radiomics preprocessing artifacts for downstream training.
        perturb_n = int(os.environ.get("SCATTERRAD_RAD_PERTURB_N", "8"))
        perturb_max_cases = int(os.environ.get("SCATTERRAD_RAD_PERTURB_MAX_CASES", "0"))
        perturb_seed = int(os.environ.get("SCATTERRAD_RAD_PERTURB_SEED", "42"))
        try:
            icc_path = compute_reproducibility_icc(
                paths=paths,
                modality=plans.modality,
                n_perturb=perturb_n,
                max_cases=perturb_max_cases,
                seed=perturb_seed,
                num_workers=workers,
            )
            log.info("Radiomics segmentation-sensitivity ICC written to %s", icc_path)
        except Exception as exc:
            log.warning("Radiomics segmentation-sensitivity analysis skipped: %s", exc)

        try:
            corr_summary = compute_intercorrelation(paths=paths, corr_threshold=0.9)
            log.info(
                "Radiomics intercorrelation summary: n_features=%s n_high_pairs=%s",
                corr_summary.get("n_features"),
                corr_summary.get("n_high_corr_pairs"),
            )
        except Exception as exc:
            log.warning("Radiomics intercorrelation analysis skipped: %s", exc)
    if do_scatter:
        from scatterrad.models.scatter.frontend import WaveletFrontend
        from scatterrad.models.scatter.scatter_cache import precompute_and_cache

        log.info("Plan requests scatter preprocessing; precomputing scattering cache.")
        frontend = WaveletFrontend(
            crop_size=tuple(plans.crop_size_voxels),
            spacing_mm=tuple(plans.target_spacing_mm),
            level=1,
            mask_mode="zero",
        )
        precompute_and_cache(paths=paths, frontend=frontend, device="cpu")


def preprocess_test(paths: ScatterRadPaths, num_workers: int = 0) -> None:
    """Preprocess test-set cases: resample, crop, and copy targets.

    Reads from imagesTs / labelsTs / targetsTs.
    Writes crops to preprocessed/crops_ts/ and targets to preprocessed/targets_ts/.
    No splits are generated.
    """
    if not paths.plans_json.exists():
        raise FileNotFoundError(
            f"{paths.plans_json} missing. Run `scatterrad plan {paths.dataset_name}` first."
        )
    if not paths.images_ts.exists() or not any(paths.images_ts.glob("*_0000.nii.gz")):
        raise FileNotFoundError(
            f"No test images found in {paths.images_ts}. "
            "Expected *_0000.nii.gz files in imagesTs/."
        )

    plans = PlansConfig.from_json(paths.plans_json)
    schema = load_targets_schema(paths.targets_json)

    basenames = sorted(p.name.replace("_0000.nii.gz", "") for p in paths.images_ts.glob("*_0000.nii.gz"))
    log.info("Preprocessing %d test cases ...", len(basenames))

    crops_ts_dir = paths.preprocessed_dataset_dir / "crops_ts"
    targets_ts_dir = paths.preprocessed_dataset_dir / "targets_ts"
    crops_ts_dir.mkdir(parents=True, exist_ok=True)
    targets_ts_dir.mkdir(parents=True, exist_ok=True)

    # Read raw target payloads for class-shift detection (reuse training shifts)
    raw_tr_payloads: dict[str, dict[str, Any]] = {}
    for basename in _case_basenames(paths.images_tr):
        raw_path = paths.targets_tr / f"{basename}.json"
        if raw_path.exists():
            raw_tr_payloads[basename] = json.loads(raw_path.read_text())

    class_shifts: dict[str, int] = {}
    for target_name, spec in schema.items():
        if spec.type is TargetType.CLASSIFICATION:
            class_shifts[target_name] = _collect_class_value_shift(
                raw_target_payloads=raw_tr_payloads,
                target_name=target_name,
                num_classes=int(spec.num_classes or 2),
            )

    # Copy + canonicalize per-case targets
    for basename in basenames:
        raw_path = paths.targets_ts / f"{basename}.json"
        if not raw_path.exists():
            log.warning("No target file for test case %s, skipping", basename)
            continue
        payload = json.loads(raw_path.read_text())
        out = dict(payload)
        for target_name, spec in schema.items():
            raw_value = _extract_scalar_value(payload, target_name)
            if spec.type is TargetType.CLASSIFICATION:
                raw_value = _normalize_class_value(
                    value=raw_value,
                    num_classes=int(spec.num_classes or 2),
                    shift=class_shifts.get(target_name, 0),
                )
            elif raw_value is not None:
                try:
                    raw_value = float(raw_value)
                except (TypeError, ValueError):
                    raw_value = None
            out[target_name] = raw_value
        (targets_ts_dir / f"{basename}.json").write_text(json.dumps(out, indent=2) + "\n")

    label_ids = tuple(int(v) for v in plans.label_coverage)
    case_args = [
        (
            str(paths.images_ts / f"{basename}_0000.nii.gz"),
            str(paths.labels_ts / f"{basename}.nii.gz"),
            basename,
            tuple(plans.target_spacing_mm),
            float(plans.crop_margin_mm),
            tuple(plans.crop_size_voxels),
            plans.modality,
            plans.intensity_clip,
            plans.intensity_mean,
            plans.intensity_std,
            label_ids,
            tuple(str(crops_ts_dir / f"{basename}_label{label_id:03d}.npz") for label_id in label_ids),
        )
        for basename in basenames
        if (paths.targets_ts / f"{basename}.json").exists()
    ]
    workers = resolve_num_workers(num_workers, max_tasks=len(case_args))
    case_flags = _run_parallel_cases(_preprocess_case, case_args, workers=workers, desc="preprocess-test")

    n_ok = sum(1 for fg, _, _ in case_flags if fg)
    for basename, (found_foreground, _wrote_any, clipped_labels) in zip(
        [a[2] for a in case_args], case_flags
    ):
        for label_id, bbox_size in clipped_labels:
            log.warning(
                "Crop clipped for test case %s label %d: bbox %s exceeds target %s",
                basename, label_id, bbox_size, tuple(plans.crop_size_voxels),
            )
    log.info("Test preprocessing done: %d/%d cases had foreground labels", n_ok, len(case_flags))
