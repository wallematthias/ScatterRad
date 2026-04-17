from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import logging
from pathlib import Path
from statistics import median

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from scatterrad.config import PlansConfig, load_dataset_config
from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.resample import resample_to_spacing
from scatterrad.utils import resolve_num_workers

log = logging.getLogger(__name__)


def _case_basenames(images_tr: Path) -> list[str]:
    return sorted(p.name.replace("_0000.nii.gz", "") for p in images_tr.glob("*_0000.nii.gz"))


def _next_multiple(v: float, base: int = 8) -> int:
    return int(np.ceil(v / base) * base)


def _plan_case(
    image_path: str,
    label_path: str,
    basename: str,
    label_ids: tuple[int, ...],
    modality: str,
) -> tuple[
    tuple[float, float, float],
    dict[int, tuple[float, float, float]],
    np.ndarray | None,
]:
    img = sitk.ReadImage(image_path)
    lbl = sitk.ReadImage(label_path)
    # BBox geometry should come from label geometry (can differ slightly from image header).
    spacing_xyz = lbl.GetSpacing()
    spacing_zyx = tuple(float(v) for v in reversed(spacing_xyz))

    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(lbl)

    bbox_sizes: dict[int, tuple[float, float, float]] = {}
    for label_id in label_ids:
        lid = int(label_id)
        if not shape.HasLabel(lid):
            continue
        bbox = shape.GetBoundingBox(lid)  # x, y, z, size_x, size_y, size_z
        size_zyx = (float(bbox[5]), float(bbox[4]), float(bbox[3]))
        bbox_sizes[label_id] = tuple(float(s * sp) for s, sp in zip(size_zyx, spacing_zyx))

    sampled_fg: np.ndarray | None = None
    if modality == "CT":
        image = sitk.GetArrayFromImage(img).astype(np.float32)
        labels = sitk.GetArrayFromImage(lbl).astype(np.int32)
        fg = labels > 0
        fg_values = image[fg]
        if fg_values.size > 10000:
            rng = np.random.default_rng(abs(hash(basename)) % (2**32))
            fg_values = rng.choice(fg_values, size=10000, replace=False)
        sampled_fg = fg_values.astype(np.float32, copy=False)
    return spacing_zyx, bbox_sizes, sampled_fg


def _resampled_bbox_case(
    label_path: str,
    target_spacing_mm: tuple[float, float, float],
    label_ids: tuple[int, ...],
) -> dict[int, tuple[int, int, int]]:
    lbl = sitk.ReadImage(label_path)
    lbl_rs = resample_to_spacing(lbl, target_spacing_mm, is_label=True)
    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(lbl_rs)
    out: dict[int, tuple[int, int, int]] = {}
    for label_id in label_ids:
        lid = int(label_id)
        if not shape.HasLabel(lid):
            continue
        bbox = shape.GetBoundingBox(lid)  # x, y, z, size_x, size_y, size_z
        out[label_id] = (int(bbox[5]), int(bbox[4]), int(bbox[3]))
    return out


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


def plan(paths: ScatterRadPaths, num_workers: int = 0, planner: str | None = "all") -> PlansConfig:
    """Read raw data, derive plans, and write plans.json."""

    ds = load_dataset_config(paths.dataset_json)
    basenames = _case_basenames(paths.images_tr)
    if not basenames:
        raise ValueError(f"No training cases in {paths.images_tr}; expected *_0000.nii.gz files")

    spacings: list[tuple[float, float, float]] = []
    bbox_by_label_mm: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    label_coverage: dict[int, int] = {lid: 0 for lid in ds.label_ids}
    sampled_fg_voxels: list[np.ndarray] = []

    workers = resolve_num_workers(num_workers, max_tasks=len(basenames))
    case_args = [
        (
            str(paths.images_tr / f"{basename}_0000.nii.gz"),
            str(paths.labels_tr / f"{basename}.nii.gz"),
            basename,
            tuple(ds.label_ids),
            ds.modality,
        )
        for basename in basenames
    ]
    results = _run_parallel_cases(_plan_case, case_args, workers=workers, desc="planning")

    for spacing_zyx, bbox_sizes, sampled_fg in results:
        spacings.append(spacing_zyx)
        for label_id, bbox_mm in bbox_sizes.items():
            label_coverage[label_id] += 1
            bbox_by_label_mm[label_id].append(bbox_mm)
        if sampled_fg is not None:
            sampled_fg_voxels.append(sampled_fg)

    target_spacing = tuple(float(median(axis)) for axis in zip(*spacings))

    bbox_percentiles: dict[int, dict[str, tuple[int, int, int]]] = {}
    for label_id in ds.label_ids:
        values = bbox_by_label_mm.get(label_id, [])
        if not values:
            bbox_percentiles[label_id] = {"p50": (0, 0, 0), "p95": (0, 0, 0)}
            log.warning("Label %s has zero coverage in dataset", label_id)
            continue
        mm = np.asarray(values, dtype=np.float32)
        p50_mm = np.percentile(mm, 50, axis=0)
        p95_mm = np.percentile(mm, 95, axis=0)
        p50_vox = np.ceil(p50_mm / np.asarray(target_spacing)).astype(int)
        p95_vox = np.ceil(p95_mm / np.asarray(target_spacing)).astype(int)
        bbox_percentiles[label_id] = {
            "p50": tuple(int(x) for x in p50_vox),
            "p95": tuple(int(x) for x in p95_vox),
        }

    # Use a robust percentile over resampled bbox sizes for fixed crop sizing.
    label_paths = [str(paths.labels_tr / f"{b}.nii.gz") for b in basenames]
    bbox_case_args = [
        (label_path, target_spacing, tuple(ds.label_ids)) for label_path in label_paths
    ]
    bbox_maps = _run_parallel_cases(
        _resampled_bbox_case, bbox_case_args, workers=workers, desc="bbox-pass"
    )
    # Compute per-label bbox distributions, then take the median-label p95 as crop size.
    # This avoids inflating crop size to the largest outlier label across the whole dataset.
    all_bbox_vox: list[np.ndarray] = []
    for bbox_map in bbox_maps:
        for bbox in bbox_map.values():
            all_bbox_vox.append(np.asarray(bbox, dtype=int))

    if all_bbox_vox:
        bbox_arr = np.vstack(all_bbox_vox).astype(np.float32)
        bbox_size_vox = np.ceil(np.percentile(bbox_arr, 99, axis=0)).astype(int)
        max_bbox_vox = bbox_arr.max(axis=0).astype(int)
        bbox_size_vox = np.maximum(bbox_size_vox, 1)
    else:
        bbox_size_vox = np.asarray([64, 64, 64], dtype=int)
        max_bbox_vox = np.asarray([64, 64, 64], dtype=int)

    margin_mm = 10.0
    margin_vox = np.ceil(margin_mm / np.asarray(target_spacing)).astype(int)
    crop_size = tuple(max(32, _next_multiple(int(v), 8)) for v in bbox_size_vox + (2 * margin_vox))
    log.info(
        "Crop size from p99 bbox + margin: p99_bbox=%s margin_vox=%s crop=%s",
        tuple(int(v) for v in bbox_size_vox),
        tuple(int(v) for v in margin_vox),
        crop_size,
    )

    intensity_clip: tuple[float, float] | None = None
    intensity_mean: float | None = None
    intensity_std: float | None = None
    if ds.modality == "CT":
        intensity_clip = (-1000.0, 1000.0)
        if sampled_fg_voxels:
            vox = np.concatenate(sampled_fg_voxels)
            vox = np.clip(vox, intensity_clip[0], intensity_clip[1])
            intensity_mean = float(vox.mean())
            intensity_std = float(vox.std())
            if intensity_std <= 0:
                intensity_std = 1.0

    plans = PlansConfig(
        version=1,
        dataset_name=ds.name,
        modality=ds.modality,
        target_spacing_mm=target_spacing,
        crop_size_voxels=crop_size,
        crop_margin_mm=margin_mm,
        intensity_clip=intensity_clip,
        intensity_mean=intensity_mean,
        intensity_std=intensity_std,
        orientation="RAS",
        label_coverage=label_coverage,
        bbox_percentiles=bbox_percentiles,
        planner=planner,
    )
    paths.ensure_preprocessed()
    plans.to_json(paths.plans_json)
    return plans
