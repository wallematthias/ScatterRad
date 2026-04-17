from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from scatterrad.models.radiomics.config import config_hash, get_pyradiomics_config
from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.crop import read_crop
from scatterrad.utils import resolve_num_workers

log = logging.getLogger(__name__)


def _suppress_radiomics_logging() -> None:
    for name in ("radiomics", "pykwalify.core", "pykwalify"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())


def _new_extractor(cfg: dict):
    _suppress_radiomics_logging()
    import radiomics
    import radiomics.featureextractor as featureextractor

    # PyRadiomics configures its own stderr handler; set quiet verbosity explicitly.
    radiomics.setVerbosity(60)
    return featureextractor.RadiomicsFeatureExtractor(cfg)


def load_features(path: Path, expected_config_hash: str) -> dict[str, float] | None:
    """Load cached features if compatible with current config hash."""

    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("config_hash") != expected_config_hash:
        return None
    features = payload.get("features", {})
    return {str(k): float(v) for k, v in features.items()}


def _to_sitk(arr: np.ndarray) -> sitk.Image:
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def extract_features_from_arrays(image: np.ndarray, mask: np.ndarray, modality: str) -> dict[str, float]:
    """Extract radiomics features from in-memory arrays."""

    if not np.any(mask):
        return {}
    cfg = get_pyradiomics_config(modality)
    extractor = _new_extractor(cfg)
    result = extractor.execute(
        _to_sitk(image.astype(np.float32)), _to_sitk(mask.astype(np.uint8)), label=1
    )

    feats: dict[str, float] = {}
    for name, value in result.items():
        if name.startswith("diagnostics_"):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        if not np.isfinite(number):
            number = float("nan")
        feats[name] = number
    return feats


def extract_case_label(
    crop_path: Path,
    output_path: Path,
    modality: str,
    force: bool = False,
) -> dict[str, float]:
    """Extract radiomics for one crop and cache results."""

    cfg = get_pyradiomics_config(modality)
    cfg_hash = config_hash(cfg)
    if not force:
        cached = load_features(output_path, cfg_hash)
        if cached is not None:
            return cached

    image, mask, _ = read_crop(crop_path)
    if not np.any(mask):
        log.warning("Skipping empty mask crop %s", crop_path.name)
        return {}
    feats = extract_features_from_arrays(image, mask, modality=modality)

    payload = {
        "config_hash": cfg_hash,
        "features": feats,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return feats


def _extract_case_label_worker(
    crop_path_str: str,
    output_path_str: str,
    modality: str,
    force: bool,
) -> tuple[str, str | None]:
    crop_path = Path(crop_path_str)
    output_path = Path(output_path_str)
    try:
        extract_case_label(crop_path, output_path, modality=modality, force=force)
        return crop_path.name, None
    except Exception as exc:
        return crop_path.name, str(exc)


def _pending_crops(
    crop_paths: list[Path],
    output_dir: Path,
    cfg_hash: str,
    force: bool,
) -> tuple[list[Path], int]:
    if force:
        return crop_paths, 0
    pending: list[Path] = []
    cached_count = 0
    for crop_path in crop_paths:
        output = output_dir / f"{crop_path.stem}.json"
        if load_features(output, cfg_hash) is None:
            pending.append(crop_path)
        else:
            cached_count += 1
    return pending, cached_count


def extract_all(
    paths: ScatterRadPaths,
    modality: str,
    num_workers: int = 0,
    force: bool = False,
    crops_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Extract features for all crops in a dataset.

    Args:
        crops_dir: override the default crops directory (e.g. for test-set crops).
        output_dir: override the default radiomics output directory.
    """

    _crops_dir = crops_dir if crops_dir is not None else paths.crops_dir
    _output_dir = output_dir if output_dir is not None else paths.radiomics_dir
    _output_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = sorted(_crops_dir.glob("*.npz"))
    cfg_hash = config_hash(get_pyradiomics_config(modality))
    pending, cached_count = _pending_crops(crop_paths, _output_dir, cfg_hash, force=force)

    if force:
        log.info("Force mode: re-extracting radiomics for %d crops.", len(pending))
    elif not pending:
        log.info("Radiomics cache hit for all %d crops; no extraction needed.", len(crop_paths))
        return
    elif cached_count == 0:
        log.info("First run extracting radiomics for %d crops.", len(pending))
    else:
        log.info(
            "Using cached radiomics for %d/%d crops; extracting %d missing crops.",
            cached_count,
            len(crop_paths),
            len(pending),
        )

    workers = resolve_num_workers(num_workers, max_tasks=len(pending))
    if workers > 1:
        jobs = [(str(crop_path), str(_output_dir / f"{crop_path.stem}.json")) for crop_path in pending]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_extract_case_label_worker, crop_path, output, modality, force)
                for crop_path, output in jobs
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="radiomics", leave=False):
                name, err = fut.result()
                if err is not None:
                    log.error("Radiomics failed for %s: %s", name, err)
    else:
        for crop_path in tqdm(pending, desc="radiomics", leave=False):
            output = _output_dir / f"{crop_path.stem}.json"
            try:
                extract_case_label(crop_path, output, modality=modality, force=force)
            except Exception as exc:
                log.error("Radiomics failed for %s: %s", crop_path.name, exc)
