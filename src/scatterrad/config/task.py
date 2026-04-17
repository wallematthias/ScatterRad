from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from pathlib import Path
from typing import Any

from .targets import TargetsSchema


class TaskConfigError(ValueError):
    """Raised when task.json is invalid."""


class ModelKind(str, Enum):
    RADIOMICS = "radiomics"
    SCATTER = "scatter"


@dataclass(frozen=True)
class CVConfig:
    folds: int = 5
    seed: int = 42


@dataclass(frozen=True)
class TaskConfig:
    """Parsed task configuration for training or inference."""

    name: str
    target: str
    model: ModelKind
    cv: CVConfig
    labels: tuple[int, ...] | None
    model_config: dict[str, Any]

    def resolved_labels(self, schema: TargetsSchema) -> tuple[int, ...]:
        if self.labels is not None:
            return self.labels
        spec = schema[self.target]
        return spec.applicable_labels


def load_task_config(path: Path, schema: TargetsSchema | None = None) -> TaskConfig:
    """Parse task.json file."""

    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise TaskConfigError(f"task config not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise TaskConfigError(f"Invalid JSON in {path}: {exc}") from exc

    name = str(data.get("name", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise TaskConfigError("task.name must match [A-Za-z0-9_-]+")

    target = str(data.get("target", "")).strip()
    if not target:
        raise TaskConfigError("task.target is required")

    if schema is not None and target not in schema:
        raise TaskConfigError(f"Unknown target '{target}'")

    try:
        model = ModelKind(str(data.get("model")))
    except ValueError as exc:
        raise TaskConfigError("task.model must be one of: radiomics, scatter") from exc

    cv_raw = data.get("cv", {})
    folds = int(cv_raw.get("folds", 5))
    seed = int(cv_raw.get("seed", 42))
    if folds < 2:
        raise TaskConfigError("cv.folds must be >=2")

    labels_raw = data.get("labels")
    labels: tuple[int, ...] | None
    if labels_raw is None:
        labels = None
    else:
        labels = tuple(sorted({int(v) for v in labels_raw}))

    model_cfg = data.get("model_config", {})
    if not isinstance(model_cfg, dict):
        raise TaskConfigError("model_config must be an object")

    return TaskConfig(
        name=name,
        target=target,
        model=model,
        cv=CVConfig(folds=folds, seed=seed),
        labels=labels,
        model_config=model_cfg,
    )
