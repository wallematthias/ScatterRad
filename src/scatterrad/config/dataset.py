from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


class DatasetConfigError(ValueError):
    """Raised when dataset.json is invalid."""


@dataclass(frozen=True)
class DatasetConfig:
    """Parsed dataset.json content."""

    name: str
    modality: str
    labels: dict[int, str]
    background_id: int

    @property
    def label_ids(self) -> list[int]:
        return sorted(self.labels)

    def label_name(self, label_id: int) -> str:
        return self.labels[label_id]


def load_dataset_config(path: Path) -> DatasetConfig:
    """Parse dataset metadata from json."""

    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise DatasetConfigError(f"dataset.json not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetConfigError(f"Invalid JSON in {path}: {exc}") from exc

    modality = data.get("modality")
    if modality is None:
        # nnU-Net style dataset.json uses "channel_names" instead of "modality".
        modality = data.get("channel_names", {})
    if not isinstance(modality, dict) or set(modality) != {"0"}:
        raise DatasetConfigError(
            "dataset.json modality must contain exactly key '0' "
            "(or provide channel_names with key '0')"
        )
    modality_name = str(modality["0"]).upper()
    if modality_name == "MRI":
        modality_name = "MR"
    if modality_name not in {"CT", "MR"}:
        raise DatasetConfigError(f"Unsupported modality '{modality_name}'")

    raw_labels = data.get("labels")
    if not isinstance(raw_labels, dict):
        raise DatasetConfigError("dataset.json labels must be an object")

    labels: dict[int, str] = {}
    for key, value in raw_labels.items():
        # ScatterRad format: {"0":"background","20":"L1"}
        # nnU-Net format: {"background":0,"T1":1}
        if isinstance(value, int):
            label_id = int(value)
            label_name = str(key)
        else:
            try:
                label_id = int(key)
            except (TypeError, ValueError) as exc:
                raise DatasetConfigError(
                    f"Label id '{key}' is not an integer and value '{value}' is not an integer"
                ) from exc
            label_name = str(value)
        labels[label_id] = label_name

    if not labels:
        raise DatasetConfigError("dataset.json must define at least one label")

    background_id = 0 if 0 in labels else -1
    for lid, name in labels.items():
        if name.lower() == "background":
            background_id = lid
            break

    foreground = {k: v for k, v in labels.items() if k != background_id}
    if not foreground:
        raise DatasetConfigError("dataset.json must include at least one foreground label")

    return DatasetConfig(
        name=str(data.get("name", path.parent.name)),
        modality=modality_name,
        labels=foreground,
        background_id=background_id,
    )
