from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from scatterrad.models.radiomics.extractor import extract_features_from_arrays
from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.crop import read_crop
from scatterrad.utils import resolve_num_workers

log = logging.getLogger(__name__)


def _shift_mask(mask: np.ndarray, dz: int, dy: int, dx: int) -> np.ndarray:
    shifted = np.roll(mask, shift=(dz, dy, dx), axis=(0, 1, 2))
    if dz > 0:
        shifted[:dz, :, :] = 0
    elif dz < 0:
        shifted[dz:, :, :] = 0
    if dy > 0:
        shifted[:, :dy, :] = 0
    elif dy < 0:
        shifted[:, dy:, :] = 0
    if dx > 0:
        shifted[:, :, :dx] = 0
    elif dx < 0:
        shifted[:, :, dx:] = 0
    return shifted


def _perturb_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    structure = np.ones((3, 3, 3), dtype=np.uint8)

    mode = rng.choice(["dilate", "erode", "shift", "open", "close"])
    if mode == "dilate":
        out = ndimage.binary_dilation(mask_u8, structure=structure, iterations=1)
    elif mode == "erode":
        out = ndimage.binary_erosion(mask_u8, structure=structure, iterations=1)
    elif mode == "open":
        out = ndimage.binary_opening(mask_u8, structure=structure, iterations=1)
    elif mode == "close":
        out = ndimage.binary_closing(mask_u8, structure=structure, iterations=1)
    else:
        dz, dy, dx = [int(v) for v in rng.integers(-1, 2, size=3)]
        out = _shift_mask(mask_u8, dz=dz, dy=dy, dx=dx)

    out_u8 = out.astype(np.uint8)
    if int(out_u8.sum()) == 0:
        return mask_u8
    return out_u8


def _icc_1_1(values: np.ndarray) -> float:
    """One-way random effects ICC(1,1) on matrix [n_cases, n_repeats]."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 2:
        return float("nan")
    n, k = x.shape
    if n < 2 or k < 2:
        return float("nan")
    if not np.isfinite(x).all():
        return float("nan")

    row_means = x.mean(axis=1, keepdims=True)
    grand_mean = float(x.mean())
    ss_between = float(k * np.sum((row_means - grand_mean) ** 2))
    ss_within = float(np.sum((x - row_means) ** 2))
    ms_between = ss_between / max(n - 1, 1)
    ms_within = ss_within / max(n * (k - 1), 1)
    denom = ms_between + (k - 1) * ms_within
    if denom <= 0:
        return float("nan")
    return float((ms_between - ms_within) / denom)


def _extract_case_perturbed(
    crop_path: Path,
    modality: str,
    n_perturb: int,
    seed: int,
) -> list[dict[str, float]]:
    image, mask, _ = read_crop(crop_path)
    if not np.any(mask):
        return []

    rng = np.random.default_rng(seed)
    feats = [extract_features_from_arrays(image, mask, modality=modality)]
    for _ in range(int(n_perturb)):
        mask_pert = _perturb_mask(mask, rng=rng)
        feats.append(extract_features_from_arrays(image, mask_pert, modality=modality))
    return feats


def _extract_case_perturbed_worker(
    crop_path_str: str,
    modality: str,
    n_perturb: int,
    seed: int,
) -> tuple[str, list[dict[str, float]] | None, str | None]:
    crop_path = Path(crop_path_str)
    try:
        feats = _extract_case_perturbed(
            crop_path=crop_path,
            modality=modality,
            n_perturb=n_perturb,
            seed=seed,
        )
        return crop_path.stem, feats, None
    except Exception as exc:
        return crop_path.stem, None, str(exc)


def compute_reproducibility_icc(
    paths: ScatterRadPaths,
    modality: str,
    n_perturb: int = 8,
    max_cases: int = 0,
    seed: int = 42,
    num_workers: int = 0,
) -> Path:
    """Compute feature ICC scores from slight mask perturbations."""

    crop_paths = sorted(paths.crops_dir.glob("*.npz"))
    if not crop_paths:
        raise ValueError(f"No crops found in {paths.crops_dir}. Run preprocess first.")
    if max_cases and max_cases > 0:
        crop_paths = crop_paths[: int(max_cases)]

    case_feature_sets: dict[str, list[dict[str, float]]] = {}
    workers = resolve_num_workers(num_workers, max_tasks=len(crop_paths))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for idx, crop_path in enumerate(crop_paths):
                local_seed = int(seed + idx * 1009)
                futures.append(
                    ex.submit(
                        _extract_case_perturbed_worker,
                        str(crop_path),
                        modality,
                        int(n_perturb),
                        local_seed,
                    )
                )
            for fut in tqdm(as_completed(futures), total=len(futures), desc="radiomics-perturb", leave=False):
                stem, feats, err = fut.result()
                if err is not None:
                    log.warning("Perturbation extraction failed for %s: %s", stem, err)
                    continue
                if feats is None or len(feats) < 2:
                    continue
                case_feature_sets[stem] = feats
    else:
        for idx, crop_path in enumerate(tqdm(crop_paths, desc="radiomics-perturb", leave=False)):
            local_seed = int(seed + idx * 1009)
            try:
                feats = _extract_case_perturbed(
                    crop_path=crop_path,
                    modality=modality,
                    n_perturb=n_perturb,
                    seed=local_seed,
                )
            except Exception as exc:
                log.warning("Perturbation extraction failed for %s: %s", crop_path.name, exc)
                continue
            if len(feats) < 2:
                continue
            case_feature_sets[crop_path.stem] = feats

    if len(case_feature_sets) < 2:
        raise ValueError("Need at least two cases with perturbation features to compute ICC")

    feature_names = sorted(
        {
            key
            for case_rows in case_feature_sets.values()
            for row in case_rows
            for key in row.keys()
            if key
        }
    )
    icc_scores: dict[str, float] = {}
    n_repeats = int(n_perturb) + 1

    for feature in feature_names:
        rows: list[np.ndarray] = []
        for case_id in sorted(case_feature_sets):
            vals = []
            for rep in case_feature_sets[case_id]:
                vals.append(float(rep.get(feature, np.nan)))
            if len(vals) != n_repeats:
                continue
            arr = np.asarray(vals, dtype=float)
            if not np.isfinite(arr).all():
                continue
            rows.append(arr)
        if len(rows) < 2:
            icc_scores[feature] = float("nan")
            continue
        x = np.vstack(rows)
        icc_scores[feature] = _icc_1_1(x)

    out_dir = paths.preprocessed_dataset_dir / "radiomics_reproducibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "icc_scores.json"
    csv_path = out_dir / "icc_scores.csv"

    json_path.write_text(json.dumps(icc_scores, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "icc"])
        for key in sorted(icc_scores):
            writer.writerow([key, icc_scores[key]])

    summary = {
        "n_cases": len(case_feature_sets),
        "n_repeats_per_case": n_repeats,
        "n_features": len(icc_scores),
        "n_features_icc_ge_0_75": int(sum(v >= 0.75 for v in icc_scores.values() if np.isfinite(v))),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log.info("Wrote radiomics reproducibility ICC to %s", json_path)
    return json_path
