from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def bbox_from_label(label_volume: np.ndarray, label_id: int) -> tuple[slice, slice, slice] | None:
    """Compute a tight bbox slice for a label, or None if absent."""

    idx = np.argwhere(label_volume == label_id)
    if idx.size == 0:
        return None
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0) + 1
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def _tight_axis_bounds(bbox_slice: slice, size: int, margin: int) -> tuple[int, int]:
    start = max(0, int(bbox_slice.start) - int(margin))
    end = min(int(size), int(bbox_slice.stop) + int(margin))
    if end <= start:
        end = min(int(size), start + 1)
    return start, end


def crop_around_bbox(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[slice, slice, slice],
    margin_voxels: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Tight crop around bbox with margin; output size varies per sample."""

    src = []
    for axis in range(3):
        s0, s1 = _tight_axis_bounds(
            bbox_slice=bbox[axis],
            size=image.shape[axis],
            margin=margin_voxels[axis],
        )
        src.append(slice(s0, s1))
    out_img = image[tuple(src)]
    out_msk = mask[tuple(src)]
    return out_img, out_msk


def write_crop(path: Path, image: np.ndarray, mask: np.ndarray, meta: dict[str, Any]) -> None:
    """Write crop payload to npz."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=image.astype(np.float32),
        mask=mask.astype(np.uint8),
        meta=np.array(meta, dtype=object),
    )


def read_crop(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load crop payload from npz."""

    with np.load(path, allow_pickle=True) as payload:
        image = payload["image"].astype(np.float32)
        mask = payload["mask"].astype(np.uint8)
        meta = payload["meta"].item()
    return image, mask, meta
