from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path


class TargetsConfigError(ValueError):
    """Raised when targets schema or case targets are invalid."""


class TargetType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TargetScope(str, Enum):
    PER_LABEL = "per_label"
    PER_CASE = "per_case"


@dataclass(frozen=True)
class TargetSpec:
    """Schema for one target entry."""

    name: str
    type: TargetType
    scope: TargetScope
    num_classes: int | None
    applicable_labels: tuple[int, ...]


class TargetsSchema(dict[str, TargetSpec]):
    """Mapping target name to specification."""

    def names(self) -> list[str]:
        return sorted(self.keys())


@dataclass(frozen=True)
class CaseTargets:
    """Per-case target values with missing values represented as NaN."""

    per_case: dict[str, float]
    per_label: dict[str, dict[int, float]]

    def get_per_case(self, name: str) -> float:
        return self.per_case.get(name, math.nan)

    def get_per_label(self, name: str, label_id: int) -> float:
        return self.per_label.get(name, {}).get(label_id, math.nan)

    def has_any_for(self, spec: TargetSpec) -> bool:
        if spec.scope is TargetScope.PER_CASE:
            return not math.isnan(self.get_per_case(spec.name))
        return any(not math.isnan(v) for v in self.per_label.get(spec.name, {}).values())


def _coerce_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_targets_schema(path: Path, known_label_ids: set[int] | None = None) -> TargetsSchema:
    """Load targets.json into validated target specs."""

    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise TargetsConfigError(f"targets.json not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise TargetsConfigError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise TargetsConfigError("targets.json must be an object")

    schema = TargetsSchema()
    for name, raw in data.items():
        if not isinstance(raw, dict):
            raise TargetsConfigError(f"Target '{name}' must be an object")
        try:
            target_type = TargetType(str(raw["type"]))
            scope = TargetScope(str(raw["scope"]))
        except KeyError as exc:
            raise TargetsConfigError(f"Target '{name}' missing key {exc}") from exc
        except ValueError as exc:
            raise TargetsConfigError(f"Target '{name}' has invalid type/scope") from exc

        num_classes: int | None = None
        if target_type is TargetType.CLASSIFICATION:
            if "num_classes" not in raw:
                raise TargetsConfigError(f"Target '{name}' requires num_classes")
            num_classes = int(raw["num_classes"])
            if num_classes < 2:
                raise TargetsConfigError(f"Target '{name}' num_classes must be >=2")
        elif "num_classes" in raw:
            raise TargetsConfigError(
                f"Target '{name}' is regression and must not define num_classes"
            )

        labels_raw = raw.get("applicable_labels", [])
        if scope is TargetScope.PER_LABEL and not labels_raw:
            raise TargetsConfigError(f"Target '{name}' per_label requires applicable_labels")
        labels = tuple(sorted({int(x) for x in labels_raw}))
        if known_label_ids is not None:
            unknown = [lid for lid in labels if lid not in known_label_ids]
            if unknown:
                raise TargetsConfigError(f"Target '{name}' has unknown labels: {unknown}")

        schema[name] = TargetSpec(
            name=name,
            type=target_type,
            scope=scope,
            num_classes=num_classes,
            applicable_labels=labels,
        )

    return schema


def load_case_targets(
    path: Path, schema: TargetsSchema, basename: str | None = None
) -> CaseTargets:
    """Load one targetsTr case file with silent dropping of unknown keys."""

    _ = basename
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise TargetsConfigError(f"targets file not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise TargetsConfigError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise TargetsConfigError(f"Case target file {path} must be an object")

    per_case: dict[str, float] = {}
    per_label: dict[str, dict[int, float]] = {}
    for name, spec in schema.items():
        value = data.get(name, None)
        if spec.scope is TargetScope.PER_CASE:
            per_case[name] = _coerce_float(value)
            continue
        label_values: dict[int, float] = {}
        if isinstance(value, dict):
            for k, v in value.items():
                try:
                    label_id = int(k)
                except (TypeError, ValueError):
                    continue
                if spec.applicable_labels and label_id not in spec.applicable_labels:
                    continue
                label_values[label_id] = _coerce_float(v)
        per_label[name] = label_values

    return CaseTargets(per_case=per_case, per_label=per_label)
