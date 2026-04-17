from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


class PlansConfigError(ValueError):
    """Raised when plans.json is invalid."""


@dataclass(frozen=True)
class PlansConfig:
    """Planner output persisted in plans.json."""

    version: int
    dataset_name: str
    modality: str
    target_spacing_mm: tuple[float, float, float]
    crop_size_voxels: tuple[int, int, int]
    crop_margin_mm: float
    intensity_clip: tuple[float, float] | None
    intensity_mean: float | None
    intensity_std: float | None
    orientation: str
    label_coverage: dict[int, int]
    bbox_percentiles: dict[int, dict[str, tuple[int, int, int]]]
    planner: str | None = None

    @classmethod
    def from_json(cls, path: Path) -> "PlansConfig":
        """Load and validate plans json."""

        try:
            data = json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise PlansConfigError(f"plans.json not found at {path}") from exc
        except json.JSONDecodeError as exc:
            raise PlansConfigError(f"Invalid JSON in {path}: {exc}") from exc

        required = {
            "version",
            "dataset_name",
            "modality",
            "target_spacing_mm",
            "crop_size_voxels",
            "crop_margin_mm",
            "intensity_clip",
            "intensity_mean",
            "intensity_std",
            "orientation",
            "label_coverage",
            "bbox_percentiles",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise PlansConfigError(f"plans.json missing keys: {missing}")

        if int(data["version"]) != 1:
            raise PlansConfigError(
                "Unsupported plans schema version. Expected version 1. "
                "Please rerun scatterrad plan to regenerate plans.json."
            )

        clip = data["intensity_clip"]
        intensity_clip: tuple[float, float] | None
        if clip is None:
            intensity_clip = None
        else:
            intensity_clip = (float(clip[0]), float(clip[1]))

        bbox_percentiles: dict[int, dict[str, tuple[int, int, int]]] = {}
        for k, entry in dict(data["bbox_percentiles"]).items():
            bbox_percentiles[int(k)] = {
                name: tuple(int(v) for v in vals) for name, vals in dict(entry).items()
            }

        return cls(
            version=1,
            dataset_name=str(data["dataset_name"]),
            modality=str(data["modality"]),
            target_spacing_mm=tuple(float(v) for v in data["target_spacing_mm"]),
            crop_size_voxels=tuple(int(v) for v in data["crop_size_voxels"]),
            crop_margin_mm=float(data["crop_margin_mm"]),
            intensity_clip=intensity_clip,
            intensity_mean=None
            if data["intensity_mean"] is None
            else float(data["intensity_mean"]),
            intensity_std=None if data["intensity_std"] is None else float(data["intensity_std"]),
            orientation=str(data["orientation"]),
            label_coverage={int(k): int(v) for k, v in dict(data["label_coverage"]).items()},
            bbox_percentiles=bbox_percentiles,
            planner=None if data.get("planner") is None else str(data.get("planner")),
        )

    def to_json(self, path: Path) -> None:
        """Write plans json in schema-compatible format."""

        payload = {
            "version": self.version,
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "target_spacing_mm": list(self.target_spacing_mm),
            "crop_size_voxels": list(self.crop_size_voxels),
            "crop_margin_mm": self.crop_margin_mm,
            "intensity_clip": list(self.intensity_clip)
            if self.intensity_clip is not None
            else None,
            "intensity_mean": self.intensity_mean,
            "intensity_std": self.intensity_std,
            "orientation": self.orientation,
            "label_coverage": {str(k): int(v) for k, v in self.label_coverage.items()},
            "bbox_percentiles": {
                str(k): {name: list(vals) for name, vals in item.items()}
                for k, item in self.bbox_percentiles.items()
            },
            "planner": self.planner,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
