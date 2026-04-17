from __future__ import annotations

import hashlib
import json


def get_pyradiomics_config(modality: str) -> dict:
    """Return modality-specific PyRadiomics configuration."""

    base = {
        "imageType": {
            "Original": {},
            "Wavelet": {},
            "LoG": {"sigma": [1.0, 2.0, 3.0]},
        },
        "featureClass": {
            "firstorder": [],
            "shape": [],
            "glcm": [],
            "glrlm": [],
            "glszm": [],
            "gldm": [],
            "ngtdm": [],
        },
        "setting": {"force2D": False},
    }
    mod = modality.upper()
    if mod == "CT":
        base["setting"].update({"binWidth": 25, "normalize": False})
    else:
        base["setting"].update({"binCount": 32, "normalize": True, "normalizeScale": 100})
    return base


def config_hash(cfg: dict) -> str:
    """Stable short hash for cache invalidation."""

    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]
