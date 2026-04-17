from __future__ import annotations

import numpy as np

from scatterrad.config import PlansConfig


def normalize(image: np.ndarray, mask: np.ndarray, plans: PlansConfig) -> np.ndarray:
    """Normalize image intensities according to modality strategy."""

    x = image.astype(np.float32, copy=True)
    fg = mask > 0
    if plans.modality == "CT":
        if plans.intensity_clip is not None:
            x = np.clip(x, plans.intensity_clip[0], plans.intensity_clip[1])
        mean = (
            plans.intensity_mean
            if plans.intensity_mean is not None
            else float(x[fg].mean() if fg.any() else x.mean())
        )
        std = (
            plans.intensity_std
            if plans.intensity_std is not None
            else float(x[fg].std() if fg.any() else x.std())
        )
        if std <= 0:
            std = 1.0
        return ((x - mean) / std).astype(np.float32)

    region = x[fg] if fg.any() else x
    mean = float(region.mean())
    std = float(region.std())
    if std <= 0:
        std = 1.0
    return ((x - mean) / std).astype(np.float32)
