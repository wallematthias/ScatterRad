from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from scatterrad.config import PlansConfig, TaskConfig, TargetScope, TargetsSchema, load_case_targets
from scatterrad.data.augment import augment_crop
from scatterrad.models.scatter.scatter_cache import load_cached_scatter
from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.crop import read_crop


class ScatterRadDataset(Dataset):
    """Dataset for scatter training in per-label or per-case scope."""

    def __init__(
        self,
        paths: ScatterRadPaths,
        basenames: list[str],
        schema: TargetsSchema,
        task: TaskConfig,
        plans: PlansConfig,
        labels: tuple[int, ...],
        augment: bool = False,
        use_scatter_cache: bool = False,
    ):
        if augment and use_scatter_cache:
            raise ValueError("augment=True cannot be combined with use_scatter_cache=True")

        self.paths = paths
        self.schema = schema
        self.task = task
        self.plans = plans
        self.labels = labels
        self.augment = augment
        self.use_scatter_cache = use_scatter_cache
        self.scope = schema[task.target].scope
        self.target_type = schema[task.target].type
        self._rng = np.random.default_rng(task.cv.seed)

        self.case_targets = {}
        for basename in basenames:
            target_file = paths.training_targets_tr / f"{basename}.json"
            if target_file.exists():
                self.case_targets[basename] = load_case_targets(
                    target_file, schema=schema, basename=basename
                )

        self.samples: list[tuple[str, int] | str] = []
        if self.scope is TargetScope.PER_LABEL:
            for basename in basenames:
                ct = self.case_targets.get(basename)
                if ct is None:
                    continue
                for label_id in labels:
                    target = ct.get_per_label(task.target, label_id)
                    if math.isnan(target):
                        continue
                    crop_path = paths.crop_path(basename, label_id)
                    if crop_path.exists():
                        self.samples.append((basename, label_id))
        else:
            for basename in basenames:
                ct = self.case_targets.get(basename)
                if ct is None:
                    continue
                target = ct.get_per_case(task.target)
                if math.isnan(target):
                    continue
                has_any = any(paths.crop_path(basename, label_id).exists() for label_id in labels)
                if has_any:
                    self.samples.append(basename)

    def __len__(self) -> int:
        return len(self.samples)

    def _target_tensor(self, value: float) -> torch.Tensor:
        if self.target_type.value == "classification":
            return torch.tensor(int(value), dtype=torch.long)
        return torch.tensor(float(value), dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if self.scope is TargetScope.PER_LABEL:
            basename, label_id = sample
            if self.use_scatter_cache:
                scatter = load_cached_scatter(self.paths, basename, label_id)
                if scatter is None:
                    raise RuntimeError(f"Scatter cache miss for {basename} label {label_id}")
                # Also load the mask so the backend can compute second-order features
                # restricted to the ROI (mask_mode applied during cache generation,
                # but GLCM needs the spatial mask at inference time).
                _, mask, _ = read_crop(self.paths.crop_path(basename, label_id))
                target_value = self.case_targets[basename].get_per_label(self.task.target, label_id)
                return {
                    "scatter": torch.from_numpy(scatter),
                    "mask": torch.from_numpy(mask[None, ...].astype(np.float32)),
                    "target": self._target_tensor(target_value),
                    "present": torch.tensor(True),
                    "meta": {"basename": basename, "label_id": int(label_id)},
                }
            image, mask, _ = read_crop(self.paths.crop_path(basename, label_id))
            if self.augment:
                image, mask = augment_crop(image, mask, self._rng)
            target_value = self.case_targets[basename].get_per_label(self.task.target, label_id)
            return {
                "image": torch.from_numpy(image[None, ...].astype(np.float32)),
                "mask": torch.from_numpy(mask[None, ...].astype(np.float32)),
                "target": self._target_tensor(target_value),
                "present": torch.tensor(True),
                "meta": {"basename": basename, "label_id": int(label_id)},
            }

        basename = sample
        imgs = []
        msks = []
        scatters = []
        present = []
        for label_id in self.labels:
            if self.use_scatter_cache:
                scatter = load_cached_scatter(self.paths, basename, label_id)
                if scatter is not None:
                    scatters.append(scatter)
                    present.append(True)
                else:
                    c, d, h, w = (self.plans.crop_size_voxels[0],) + (1, 1, 1)  # placeholder shape
                    # We don't know out_channels here; will be filled with zeros below
                    scatters.append(None)
                    present.append(False)
            else:
                crop_path = self.paths.crop_path(basename, label_id)
                if crop_path.exists():
                    image, mask, _ = read_crop(crop_path)
                    if self.augment:
                        image, mask = augment_crop(image, mask, self._rng)
                    present.append(True)
                else:
                    image = np.zeros(self.plans.crop_size_voxels, dtype=np.float32)
                    mask = np.zeros(self.plans.crop_size_voxels, dtype=np.uint8)
                    present.append(False)
                imgs.append(image)
                msks.append(mask)

        target_value = self.case_targets[basename].get_per_case(self.task.target)

        if self.use_scatter_cache:
            # Fill missing labels with zeros matching the shape of the first present cache entry
            ref = next((s for s in scatters if s is not None), None)
            if ref is None:
                raise RuntimeError(f"No scatter cache entries found for {basename}")
            filled = [s if s is not None else np.zeros_like(ref) for s in scatters]
            return {
                "scatter": torch.from_numpy(np.stack(filled, axis=0).astype(np.float32)),
                "target": self._target_tensor(target_value),
                "present": torch.tensor(present, dtype=torch.bool),
                "meta": {"basename": basename, "label_ids": list(self.labels)},
            }

        return {
            "images": torch.from_numpy(np.asarray(imgs, dtype=np.float32)[:, None, ...]),
            "masks": torch.from_numpy(np.asarray(msks, dtype=np.float32)[:, None, ...]),
            "target": self._target_tensor(target_value),
            "present": torch.tensor(present, dtype=torch.bool),
            "meta": {"basename": basename, "label_ids": list(self.labels)},
        }
