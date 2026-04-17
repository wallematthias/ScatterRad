from __future__ import annotations

import numpy as np
from scipy import ndimage


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


def augment_crop(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply lightweight geometric augmentation with deterministic RNG input."""

    x = image.copy()
    m = mask.copy()

    for axis in range(3):
        if rng.random() < 0.5:
            x = np.flip(x, axis=axis)
            m = np.flip(m, axis=axis)

    if rng.random() < 0.5:
        axes_options = [(0, 1), (0, 2), (1, 2)]
        axes = axes_options[int(rng.integers(0, len(axes_options)))]
        angle = float(rng.uniform(-10.0, 10.0))
        x = ndimage.rotate(x, angle=angle, axes=axes, reshape=False, order=3, mode="nearest")
        m = ndimage.rotate(m, angle=angle, axes=axes, reshape=False, order=0, mode="nearest")

    if rng.random() < 0.5:
        scale = float(rng.uniform(0.95, 1.05))
        x = ndimage.zoom(x, zoom=scale, order=3)
        m = ndimage.zoom(m, zoom=scale, order=0)
        x = _center_crop_or_pad(x, image.shape, order=3)
        m = _center_crop_or_pad(m, mask.shape, order=0)

    return x.astype(np.float32), (m > 0.5).astype(np.uint8)
