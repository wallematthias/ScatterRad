from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scatterrad.models.scatter.frontend import ScatterFrontend
from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.crop import read_crop


def _cache_key(frontend: ScatterFrontend) -> str:
    payload = {
        "wavelet": getattr(frontend, "wavelet", None),
        "level": getattr(frontend, "level", None),
        "J": frontend.J,
        "L": frontend.L,
        "mask_mode": frontend.mask_mode,
        "crop_size": frontend.crop_size,
        "log_sigmas_mm": getattr(frontend, "log_sigmas_mm", None),
        "use_gradient": getattr(frontend, "use_gradient", None),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def _cache_dir(paths: ScatterRadPaths, frontend: ScatterFrontend) -> Path:
    _ = frontend
    return paths.preprocessed_dataset_dir / "scatter_cache"


def _process_one(
    crop_path: str,
    out_path: str,
    expected_shape: tuple,
    frontend_kwargs: dict,
) -> str:
    """Worker: compute filter-bank features for one crop and save to .npy.

    Runs in a subprocess — imports are local to avoid pickling issues.
    Returns the crop stem on success.
    """
    import numpy as np
    from scatterrad.models.scatter.frontend import WaveletFrontend
    from scatterrad.preprocessing.crop import read_crop
    import torch

    image, mask, _ = read_crop(Path(crop_path))
    if tuple(int(v) for v in image.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"Crop shape mismatch for {Path(crop_path).name}: "
            f"got {tuple(image.shape)}, expected {tuple(expected_shape)}. "
            "Rerun `scatterrad preprocess` to regenerate stale crops."
        )

    fe = WaveletFrontend(**frontend_kwargs)
    x = torch.from_numpy(image[None, None, ...]).float()
    m = torch.from_numpy(mask[None, None, ...]).float()
    with torch.no_grad():
        out, _ = fe(x, m)
    np.save(out_path, out.squeeze(0).numpy().astype(np.float32))
    return Path(crop_path).stem


def precompute_and_cache(
    paths: ScatterRadPaths,
    frontend: ScatterFrontend,
    device: str = "cpu",
) -> None:
    """Precompute filter-bank outputs for all crops, in parallel."""

    cdir = _cache_dir(paths, frontend)
    cdir.mkdir(parents=True, exist_ok=True)

    key_path = cdir / "cache_key.txt"
    current_key = _cache_key(frontend)
    if key_path.exists():
        existing_key = key_path.read_text().strip()
        if existing_key and existing_key != current_key:
            for stale in cdir.glob("*.npy"):
                stale.unlink()
    key_path.write_text(current_key + "\n")

    crop_paths = sorted(paths.crops_dir.glob("*.npz"))
    pending = [p for p in crop_paths if not (cdir / f"{p.stem}.npy").exists()]

    if not pending:
        return

    # Serialisable kwargs for the worker (no nn.Module pickling)
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

    args_list = [
        (str(p), str(cdir / f"{p.stem}.npy"), tuple(frontend.crop_size), frontend_kwargs)
        for p in pending
    ]

    if num_workers <= 1:
        for args in tqdm(args_list, desc="scatter-cache"):
            _process_one(*args)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_process_one, *args): args[0] for args in args_list}
        with tqdm(total=len(futures), desc=f"scatter-cache ({num_workers} workers)") as pbar:
            for fut in as_completed(futures):
                fut.result()  # re-raises any exception from the worker
                pbar.update(1)


def load_cached_scatter(paths: ScatterRadPaths, basename: str, label_id: int) -> np.ndarray | None:
    """Load cached filter-bank output when available."""
    cache_file = paths.preprocessed_dataset_dir / "scatter_cache" / f"{basename}_label{label_id:03d}.npy"
    if not cache_file.exists():
        return None
    return np.load(cache_file).astype(np.float32)
