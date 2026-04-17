from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re


class ScatterRadPathError(ValueError):
    """Raised when path configuration is invalid."""


@dataclass(frozen=True)
class ScatterRadPaths:
    """Environment-variable driven path resolver for one dataset."""

    dataset_name: str
    raw_root: Path | None
    preprocessed_root: Path
    results_root: Path

    @classmethod
    def from_env(cls, dataset_name: str, require_raw: bool = True) -> "ScatterRadPaths":
        raw = os.environ.get("SCATTERRAD_RAW")
        pre = os.environ.get("SCATTERRAD_PREPROCESSED")
        res = os.environ.get("SCATTERRAD_RESULTS")
        missing = []
        if require_raw and not raw:
            missing.append("SCATTERRAD_RAW")
        if not pre:
            missing.append("SCATTERRAD_PREPROCESSED")
        if not res:
            missing.append("SCATTERRAD_RESULTS")
        if missing:
            missing_str = ", ".join(missing)
            raise ScatterRadPathError(f"Missing required environment variables: {missing_str}")
        return cls(
            dataset_name=dataset_name,
            raw_root=Path(raw) if raw else None,
            preprocessed_root=Path(pre),
            results_root=Path(res),
        )

    @property
    def raw_dataset_dir(self) -> Path:
        if self.raw_root is None:
            raise ScatterRadPathError("SCATTERRAD_RAW unavailable for this workflow")
        return self.raw_root / self.dataset_name

    @property
    def preprocessed_dataset_dir(self) -> Path:
        return self.preprocessed_root / self.dataset_name

    @property
    def results_dataset_dir(self) -> Path:
        return self.results_root / self.dataset_name

    @property
    def dataset_json(self) -> Path:
        return self.raw_dataset_dir / "dataset.json"

    @property
    def targets_json(self) -> Path:
        return self.raw_dataset_dir / "targets.json"

    @property
    def images_tr(self) -> Path:
        return self.raw_dataset_dir / "imagesTr"

    @property
    def labels_tr(self) -> Path:
        return self.raw_dataset_dir / "labelsTr"

    @property
    def targets_tr(self) -> Path:
        return self.raw_dataset_dir / "targetsTr"

    @property
    def images_ts(self) -> Path:
        return self.raw_dataset_dir / "imagesTs"

    @property
    def labels_ts(self) -> Path:
        return self.raw_dataset_dir / "labelsTs"

    @property
    def targets_ts(self) -> Path:
        return self.raw_dataset_dir / "targetsTs"

    @property
    def preprocessed_dataset_json(self) -> Path:
        return self.preprocessed_dataset_dir / "dataset.json"

    @property
    def preprocessed_targets_json(self) -> Path:
        return self.preprocessed_dataset_dir / "targets.json"

    @property
    def preprocessed_targets_tr(self) -> Path:
        return self.preprocessed_dataset_dir / "targetsTr"

    @property
    def training_dataset_json(self) -> Path:
        """Dataset metadata used for train-time workflows."""

        if self.preprocessed_dataset_json.exists():
            return self.preprocessed_dataset_json
        return self.dataset_json

    @property
    def training_targets_json(self) -> Path:
        """Target schema used for train-time workflows."""

        if self.preprocessed_targets_json.exists():
            return self.preprocessed_targets_json
        return self.targets_json

    @property
    def training_targets_tr(self) -> Path:
        """Per-case targets used for train-time workflows."""

        if self.preprocessed_targets_tr.exists():
            return self.preprocessed_targets_tr
        return self.targets_tr

    @property
    def plans_json(self) -> Path:
        return self.preprocessed_dataset_dir / "plans.json"

    @property
    def splits_json(self) -> Path:
        return self.preprocessed_dataset_dir / "splits.json"

    @property
    def results_splits_json(self) -> Path:
        return self.results_dataset_dir / "splits.json"

    @property
    def crops_dir(self) -> Path:
        return self.preprocessed_dataset_dir / "crops"

    @property
    def radiomics_dir(self) -> Path:
        return self.preprocessed_dataset_dir / "radiomics"

    @property
    def radiomics_ts_dir(self) -> Path:
        return self.preprocessed_dataset_dir / "radiomics_ts"

    @property
    def crops_ts_dir(self) -> Path:
        return self.preprocessed_dataset_dir / "crops_ts"

    def crop_path(self, basename: str, label_id: int) -> Path:
        return self.crops_dir / f"{basename}_label{label_id:03d}.npz"

    def radiomics_path(self, basename: str, label_id: int) -> Path:
        return self.radiomics_dir / f"{basename}_label{label_id:03d}.json"

    @staticmethod
    def _safe_component(name: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", name)
        return cleaned or fallback

    @staticmethod
    def _planner_dir_name(planner_name: str | None, model_kind: str | None = None) -> str:
        if planner_name is not None and planner_name.strip():
            raw = planner_name.strip()
            if raw.lower() in {"all", "both"}:
                return "AllPlanner"
            if raw.lower().endswith("planner"):
                return ScatterRadPaths._safe_component(raw, "AllPlanner")
            return ScatterRadPaths._safe_component(f"{raw}Planner", "AllPlanner")
        if model_kind == "radiomics":
            return "RadiomicsPlanner"
        if model_kind == "scatter":
            return "ScatterPlanner"
        return "AllPlanner"

    def _legacy_result_dir(self, task_name: str, model_kind: str, fold: int) -> Path:
        return self.results_dataset_dir / f"{task_name}__{model_kind}__fold{fold}"

    def result_config_dir(
        self,
        task_name: str,
        model_kind: str | None = None,
        planner_name: str | None = None,
        trainer_name: str = "scatterradDefaultTrainer",
    ) -> Path:
        trainer = self._safe_component(trainer_name, "scatterradDefaultTrainer")
        planner = self._planner_dir_name(planner_name, model_kind=model_kind)
        return self.results_dataset_dir / task_name / f"{trainer}__{planner}"

    def result_dir(
        self,
        task_name: str,
        model_kind: str,
        fold: int,
        planner_name: str | None = None,
        trainer_name: str = "scatterradDefaultTrainer",
    ) -> Path:
        """Resolve fold result directory.

        If planner_name is provided, return nnUNet-style path:
        results/<dataset>/<task>/<trainer>__<planner>/fold<k>
        For read-time calls (planner_name is None), auto-detect existing nnUNet-style
        runs and fall back to legacy flat directories.
        """

        if planner_name is not None:
            return (
                self.result_config_dir(
                    task_name,
                    model_kind=model_kind,
                    planner_name=planner_name,
                    trainer_name=trainer_name,
                )
                / f"fold{fold}"
            )

        # Legacy paths from older runs.
        legacy = self._legacy_result_dir(task_name, model_kind, fold)
        if legacy.exists():
            return legacy

        # Prefer any existing nnUNet-like layout for this task/fold.
        task_root = self.results_dataset_dir / task_name
        if task_root.exists():
            candidates = sorted(
                p for p in task_root.glob(f"*/*fold{fold}") if p.is_dir() and p.name == f"fold{fold}"
            )
            if candidates:
                return candidates[0]

        # Default write target when caller does not specify planner.
        return (
            self.result_config_dir(
                task_name,
                model_kind=model_kind,
                planner_name=None,
                trainer_name=trainer_name,
            )
            / f"fold{fold}"
        )

    def result_fold_dirs(self, task_name: str, model_kind: str) -> list[Path]:
        """List fold directories across both new and legacy layouts."""

        dirs: list[Path] = []
        task_root = self.results_dataset_dir / task_name
        if task_root.exists():
            for p in sorted(task_root.glob("*/*")):
                if not p.is_dir() or not p.name.startswith("fold"):
                    continue
                metrics_file = p / "metrics.json"
                if metrics_file.exists():
                    try:
                        payload = json.loads(metrics_file.read_text())
                        model = payload.get("model")
                    except json.JSONDecodeError:
                        model = None
                    if isinstance(model, str) and model != model_kind:
                        continue
                dirs.append(p)
        legacy_prefix = f"{task_name}__{model_kind}__fold"
        dirs.extend(
            sorted(p for p in self.results_dataset_dir.glob(f"{legacy_prefix}*") if p.is_dir())
        )
        # Deduplicate while preserving order.
        seen = set()
        unique: list[Path] = []
        for item in dirs:
            key = str(item.resolve()) if item.exists() else str(item)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def ensure_preprocessed(self) -> Path:
        self.preprocessed_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.radiomics_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_targets_tr.mkdir(parents=True, exist_ok=True)
        return self.preprocessed_dataset_dir

    def ensure_results(self) -> Path:
        self.results_dataset_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dataset_dir
