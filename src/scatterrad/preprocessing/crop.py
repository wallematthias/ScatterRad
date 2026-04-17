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


def _axis_bounds(
    bbox_slice: slice,
    size: int,
    target: int,
    margin: int,
) -> tuple[int, int, int, int, bool]:
    low = bbox_slice.start
    high = bbox_slice.stop
    bbox_size = high - low
    clipped = False
    if target < bbox_size:
        # Bbox exceeds target (outlier case) — center the crop and let it clip.
        center = (low + high) // 2
        low = center - target // 2
        high = low + target
        bbox_size = target
        clipped = True

    # If the requested context does not fit, reduce margin so ROI still fits target.
    max_margin = max(0, (target - bbox_size) // 2)
    effective_margin = min(margin, max_margin)
    expanded_low = low - effective_margin
    expanded_high = high + effective_margin

    center = (expanded_low + expanded_high) // 2
    start = center - (target // 2)
    end = start + target
    if start > expanded_low:
        start = expanded_low
        end = start + target
    if end < expanded_high:
        end = expanded_high
        start = end - target

    src_start = max(start, 0)
    src_end = min(end, size)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    return src_start, src_end, dst_start, dst_end, clipped


def crop_around_bbox(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[slice, slice, slice],
    target_size: tuple[int, int, int],
    margin_voxels: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Center crop around bbox and pad as needed to fixed target size.

    Returns (cropped_image, cropped_mask, was_clipped) where was_clipped is True
    if the bbox exceeded the target size in any axis.
    """

    out_img = np.zeros(target_size, dtype=image.dtype)
    out_msk = np.zeros(target_size, dtype=mask.dtype)

    src = []
    dst = []
    any_clipped = False
    for axis in range(3):
        s0, s1, d0, d1, clipped = _axis_bounds(
            bbox[axis],
            image.shape[axis],
            target_size[axis],
            margin_voxels[axis],
        )
        src.append(slice(s0, s1))
        dst.append(slice(d0, d1))
        any_clipped = any_clipped or clipped

    out_img[tuple(dst)] = image[tuple(src)]
    out_msk[tuple(dst)] = mask[tuple(src)]
    return out_img, out_msk, any_clipped


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
