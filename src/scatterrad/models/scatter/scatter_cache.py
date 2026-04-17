from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from scatterrad.models.scatter.frontend import ScatterFrontend
from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.crop import read_crop

_CACHE_STEM_RE = re.compile(r"^(?P<basename>.+)_label(?P<label_id>\d{3})(?:_aug(?P<variant>\d{3}))?$")


def _center_crop_or_pad(volume: np.ndarray, shape: tuple[int, int, int], order: int) -> np.ndarray:
    out = np.zeros(shape, dtype=volume.dtype)
    src_slices = []
    dst_slices = []
    for dim, target in zip(volume.shape, shape):
        if dim >= target:
            start = (dim - target) // 2
            src_slices.append(slice(start, start + target))
            dst_slices.append(slice(0, target))
        else:
            pad = (target - dim) // 2
            src_slices.append(slice(0, dim))
            dst_slices.append(slice(pad, pad + dim))
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out


def _augment_crop_for_cache(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    intensity_scale_delta: float = 0.1,
    intensity_shift_delta: float = 0.1,
    noise_std: float = 0.05,
    elastic_alpha: float = 1.0,
    elastic_sigma: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    x = image.astype(np.float32, copy=True)
    m = mask.astype(np.uint8, copy=True)

    # --- Geometric augmentations (applied to raw image before filter-bank computation) ---

    # Random rotation (one axis-pair, ±10°).
    if rng.random() < 0.5:
        axes_options = [(0, 1), (0, 2), (1, 2)]
        axes = axes_options[int(rng.integers(0, len(axes_options)))]
        angle = float(rng.uniform(-10.0, 10.0))
        x = ndimage.rotate(x, angle=angle, axes=axes, reshape=False, order=3, mode="nearest")
        m = ndimage.rotate(m, angle=angle, axes=axes, reshape=False, order=0, mode="nearest")

    # Random isotropic zoom (±5%), re-centred to original size.
    if rng.random() < 0.5:
        orig_shape = x.shape
        scale = float(rng.uniform(0.95, 1.05))
        x = ndimage.zoom(x, zoom=scale, order=3)
        m = ndimage.zoom(m, zoom=scale, order=0)
        x = _center_crop_or_pad(x, orig_shape, order=3)
        m = _center_crop_or_pad(m, orig_shape, order=0)

    # --- Intensity augmentations ---

    scale = float(rng.uniform(1.0 - intensity_scale_delta, 1.0 + intensity_scale_delta))
    shift = float(rng.uniform(-intensity_shift_delta, intensity_shift_delta))
    x = x * scale + shift

    if noise_std > 0:
        sigma = float(rng.uniform(0.0, noise_std))
        if sigma > 0:
            x = x + rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)

    # --- Elastic deformation ---

    if elastic_alpha > 0 and elastic_sigma > 0:
        shape = x.shape
        coords = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            np.arange(shape[2], dtype=np.float32),
            indexing="ij",
        )
        disp = []
        for _axis in range(3):
            delta = rng.standard_normal(shape).astype(np.float32)
            delta = ndimage.gaussian_filter(delta, sigma=elastic_sigma, mode="reflect")
            disp.append(delta * float(elastic_alpha))
        sample_coords = [base + delta for base, delta in zip(coords, disp)]
        x = ndimage.map_coordinates(x, sample_coords, order=3, mode="nearest")
        m = ndimage.map_coordinates(m.astype(np.float32), sample_coords, order=0, mode="nearest")

    return x.astype(np.float32, copy=False), (m > 0.5).astype(np.uint8, copy=False)


def _cache_key(frontend: ScatterFrontend, cache_aug_config: dict[str, int | float]) -> str:
    cache_dtype = os.environ.get("SCATTERRAD_SCATTER_CACHE_DTYPE", "float16").strip().lower()
    payload = {
        "wavelet": getattr(frontend, "wavelet", None),
        "level": getattr(frontend, "level", None),
        "J": frontend.J,
        "L": frontend.L,
        "mask_mode": frontend.mask_mode,
        "crop_size": frontend.crop_size,
        "log_sigmas_mm": getattr(frontend, "log_sigmas_mm", None),
        "use_gradient": getattr(frontend, "use_gradient", None),
        "cache_dtype": cache_dtype,
        "cache_augment": cache_aug_config,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def _cache_dir(paths: ScatterRadPaths, frontend: ScatterFrontend) -> Path:
    _ = frontend
    return paths.preprocessed_dataset_dir / "scatter_cache"


def _cache_stem(basename: str, label_id: int, variant_idx: int = 0) -> str:
    stem = f"{basename}_label{label_id:03d}"
    if variant_idx > 0:
        stem = f"{stem}_aug{variant_idx:03d}"
    return stem


def scatter_cache_path(paths: ScatterRadPaths, basename: str, label_id: int, variant_idx: int = 0) -> Path:
    return paths.preprocessed_dataset_dir / "scatter_cache" / f"{_cache_stem(basename, label_id, variant_idx)}.npy"


def scatter_cache_paths(
    paths: ScatterRadPaths,
    basename: str,
    label_id: int,
    num_augmented_variants: int = 0,
) -> list[Path]:
    return [
        scatter_cache_path(paths, basename, label_id, variant_idx=variant_idx)
        for variant_idx in range(0, int(num_augmented_variants) + 1)
    ]


def parse_scatter_cache_filename(path: Path) -> tuple[str, int, int] | None:
    match = _CACHE_STEM_RE.match(path.stem)
    if match is None:
        return None
    basename = str(match.group("basename"))
    label_id = int(match.group("label_id"))
    variant_idx = int(match.group("variant") or 0)
    return basename, label_id, variant_idx


def _variant_seed(crop_stem: str, variant_idx: int, base_seed: int) -> int:
    payload = f"{crop_stem}:{variant_idx}:{base_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False)


def _process_one(
    crop_path: str,
    out_path: str,
    frontend_kwargs: dict,
    out_dtype_str: str,
    variant_idx: int,
    cache_aug_config: dict[str, int | float],
) -> str:
    """Worker: compute filter-bank features for one crop variant and save to .npy."""
    import numpy as np
    import torch

    from scatterrad.models.scatter.frontend import WaveletFrontend
    from scatterrad.preprocessing.crop import read_crop

    image, mask, _ = read_crop(Path(crop_path))
    if variant_idx > 0:
        rng = np.random.default_rng(
            _variant_seed(Path(crop_path).stem, variant_idx, int(cache_aug_config["seed"]))
        )
        image, mask = _augment_crop_for_cache(
            image,
            mask,
            rng,
            intensity_scale_delta=float(cache_aug_config["intensity_scale_delta"]),
            intensity_shift_delta=float(cache_aug_config["intensity_shift_delta"]),
            noise_std=float(cache_aug_config["noise_std"]),
            elastic_alpha=float(cache_aug_config["elastic_alpha"]),
            elastic_sigma=float(cache_aug_config["elastic_sigma"]),
        )
    fe = WaveletFrontend(**frontend_kwargs)
    x = torch.from_numpy(image[None, None, ...]).float()
    m = torch.from_numpy(mask[None, None, ...].astype(np.float32)).float()
    with torch.no_grad():
        out, _ = fe(x, m)
    out_dtype = np.float16 if out_dtype_str == "float16" else np.float32
    np.save(out_path, out.squeeze(0).numpy().astype(out_dtype, copy=False))
    return Path(out_path).stem


def precompute_and_cache(
    paths: ScatterRadPaths,
    frontend: ScatterFrontend,
    device: str = "cpu",
    *,
    num_augmented_variants: int = 0,
    cache_aug_seed: int = 42,
    intensity_scale_delta: float = 0.1,
    intensity_shift_delta: float = 0.1,
    noise_std: float = 0.05,
    elastic_alpha: float = 1.0,
    elastic_sigma: float = 6.0,
) -> None:
    """Precompute filter-bank outputs for all crops, optionally with augmented variants."""

    _ = device
    cdir = _cache_dir(paths, frontend)
    cdir.mkdir(parents=True, exist_ok=True)

    cache_aug_config: dict[str, int | float] = {
        "num_augmented_variants": int(num_augmented_variants),
        "seed": int(cache_aug_seed),
        "intensity_scale_delta": float(intensity_scale_delta),
        "intensity_shift_delta": float(intensity_shift_delta),
        "noise_std": float(noise_std),
        "elastic_alpha": float(elastic_alpha),
        "elastic_sigma": float(elastic_sigma),
    }

    key_path = cdir / "cache_key.txt"
    current_key = _cache_key(frontend, cache_aug_config)
    if key_path.exists():
        existing_key = key_path.read_text().strip()
        if existing_key and existing_key != current_key:
            for stale in cdir.glob("*.npy"):
                stale.unlink()
    key_path.write_text(current_key + "\n")

    crop_paths = sorted(paths.crops_dir.glob("*.npz"))
    pending: list[tuple[Path, int]] = []
    for crop_path in crop_paths:
        parsed = parse_scatter_cache_filename(Path(f"{crop_path.stem}.npy"))
        if parsed is None:
            continue
        basename, label_id, _ = parsed
        for variant_idx in range(0, int(num_augmented_variants) + 1):
            cache_path = scatter_cache_path(paths, basename, label_id, variant_idx=variant_idx)
            if not cache_path.exists():
                pending.append((crop_path, variant_idx))

    if not pending:
        return

    frontend_kwargs = dict(
        crop_size=tuple(frontend.crop_size),
        spacing_mm=tuple(getattr(frontend, "spacing_mm", (1.0, 1.0, 1.0))),
        wavelet=getattr(frontend, "wavelet", "coif1"),
        level=int(getattr(frontend, "level", 1)),
        log_sigmas_mm=tuple(getattr(frontend, "log_sigmas_mm", (1.0, 2.0, 3.0))),
        use_gradient=bool(getattr(frontend, "use_gradient", True)),
        mask_mode=str(frontend.mask_mode),
    )

    num_workers = int(os.environ.get("SCATTERRAD_NP", 0)) or os.cpu_count() or 1

    out_dtype_str = os.environ.get("SCATTERRAD_SCATTER_CACHE_DTYPE", "float16").strip().lower()
    if out_dtype_str not in {"float16", "float32"}:
        out_dtype_str = "float16"
    args_list = []
    for crop_path, variant_idx in pending:
        basename, label_id, _ = parse_scatter_cache_filename(Path(f"{crop_path.stem}.npy")) or ("", 0, 0)
        args_list.append(
            (
                str(crop_path),
                str(scatter_cache_path(paths, basename, label_id, variant_idx=variant_idx)),
                frontend_kwargs,
                out_dtype_str,
                variant_idx,
                cache_aug_config,
            )
        )

    if num_workers <= 1:
        for args in tqdm(args_list, desc="scatter-cache"):
            _process_one(*args)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_process_one, *args): args[0] for args in args_list}
        with tqdm(total=len(futures), desc=f"scatter-cache ({num_workers} workers)") as pbar:
            for fut in as_completed(futures):
                fut.result()
                pbar.update(1)


def load_cached_scatter(
    paths: ScatterRadPaths,
    basename: str,
    label_id: int,
    variant_idx: int = 0,
) -> np.ndarray | None:
    """Load cached filter-bank output when available."""
    cache_file = scatter_cache_path(paths, basename, label_id, variant_idx=variant_idx)
    if not cache_file.exists():
        return None
    arr = np.load(cache_file, mmap_mode="r")
    return np.asarray(arr, dtype=np.float32)
