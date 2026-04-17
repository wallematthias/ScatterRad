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
        scatter_cache_tensors: dict[tuple[str, int], torch.Tensor] | None = None,
        scatter_mask_tensors: dict[tuple[str, int], torch.Tensor] | None = None,
    ):
        self.paths = paths
        self.schema = schema
        self.task = task
        self.plans = plans
        self.labels = labels
        self.augment = augment
        self.use_scatter_cache = use_scatter_cache
        self._need_cache_mask = bool(task.model_config.get("second_order", False))
        self.scatter_cache_tensors = scatter_cache_tensors or {}
        self.scatter_mask_tensors = scatter_mask_tensors or {}
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

    @staticmethod
    def _pad_numpy_to_shape(arr: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=arr.dtype)
        d, h, w = arr.shape
        out[:d, :h, :w] = arr
        return out

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if self.scope is TargetScope.PER_LABEL:
            basename, label_id = sample
            if self.use_scatter_cache:
                key = (basename, int(label_id))
                scatter_tensor = self.scatter_cache_tensors.get(key)
                if scatter_tensor is None:
                    scatter = load_cached_scatter(self.paths, basename, label_id)
                    if scatter is None:
                        raise RuntimeError(f"Scatter cache miss for {basename} label {label_id}")
                    scatter_tensor = torch.from_numpy(scatter)
                mask_tensor = None
                if self._need_cache_mask:
                    mask_tensor = self.scatter_mask_tensors.get(key)
                    if mask_tensor is None:
                        _, mask, _ = read_crop(self.paths.crop_path(basename, label_id))
                        mask_tensor = torch.from_numpy(mask[None, ...].astype(np.float32))
                if self.augment:
                    scatter_tensor, mask_tensor = self._augment_cached_sample(
                        scatter_tensor, mask_tensor
                    )
                target_value = self.case_targets[basename].get_per_label(self.task.target, label_id)
                item = {
                    "scatter": scatter_tensor,
                    "target": self._target_tensor(target_value),
                    "present": torch.tensor(True),
                    "meta": {"basename": basename, "label_id": int(label_id)},
                }
                if self._need_cache_mask:
                    item["mask"] = mask_tensor
                return item

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
                key = (basename, int(label_id))
                scatter_tensor = self.scatter_cache_tensors.get(key)
                if scatter_tensor is not None:
                    if self.augment:
                        scatter_tensor, _ = self._augment_cached_sample(scatter_tensor, None)
                    scatters.append(scatter_tensor)
                    present.append(True)
                else:
                    scatter = load_cached_scatter(self.paths, basename, label_id)
                    if scatter is not None:
                        scatter_tensor = torch.from_numpy(scatter)
                        if self.augment:
                            scatter_tensor, _ = self._augment_cached_sample(scatter_tensor, None)
                        scatters.append(scatter_tensor)
                        present.append(True)
                    else:
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
                    image = None
                    mask = None
                    present.append(False)
                imgs.append(image)
                msks.append(mask)

        target_value = self.case_targets[basename].get_per_case(self.task.target)

        if self.use_scatter_cache:
            # Fill missing labels with zeros matching the shape of the first present cache entry
            ref = next((s for s in scatters if s is not None), None)
            if ref is None:
                raise RuntimeError(f"No scatter cache entries found for {basename}")
            filled = [s if s is not None else torch.zeros_like(ref) for s in scatters]
            scatter = torch.stack(filled, dim=0).float()
            return {
                "scatter": scatter,
                "target": self._target_tensor(target_value),
                "present": torch.tensor(present, dtype=torch.bool),
                "meta": {"basename": basename, "label_ids": list(self.labels)},
            }

        present_imgs = [x for x in imgs if x is not None]
        if not present_imgs:
            raise RuntimeError(f"No image crops found for {basename}")
        max_d = max(int(x.shape[0]) for x in present_imgs)
        max_h = max(int(x.shape[1]) for x in present_imgs)
        max_w = max(int(x.shape[2]) for x in present_imgs)
        target_shape = (max_d, max_h, max_w)
        imgs_pad = [
            self._pad_numpy_to_shape(x, target_shape)
            if x is not None
            else np.zeros(target_shape, dtype=np.float32)
            for x in imgs
        ]
        msks_pad = [
            self._pad_numpy_to_shape(x, target_shape)
            if x is not None
            else np.zeros(target_shape, dtype=np.uint8)
            for x in msks
        ]
        return {
            "images": torch.from_numpy(np.asarray(imgs_pad, dtype=np.float32)[:, None, ...]),
            "masks": torch.from_numpy(np.asarray(msks_pad, dtype=np.float32)[:, None, ...]),
            "target": self._target_tensor(target_value),
            "present": torch.tensor(present, dtype=torch.bool),
            "meta": {"basename": basename, "label_ids": list(self.labels)},
        }

    def _augment_cached_sample(
        self,
        scatter: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Lightweight augmentation for cached filter-bank tensors.

        Works with scatter shape (C,D,H,W). Applies random flips and small integer shifts.
        """
        s = scatter
        m = mask
        # Random flips along spatial axes.
        for axis in (-3, -2, -1):
            if self._rng.random() < 0.5:
                s = torch.flip(s, dims=(axis,))
                if m is not None:
                    m = torch.flip(m, dims=(axis,))

        # Random small translation.
        if self._rng.random() < 0.5:
            shifts = [int(v) for v in self._rng.integers(-2, 3, size=3)]
            s = torch.roll(s, shifts=tuple(shifts), dims=(-3, -2, -1))
            if m is not None:
                m = torch.roll(m, shifts=tuple(shifts), dims=(-3, -2, -1))
            # Zero-fill wrapped regions after roll.
            dz, dy, dx = shifts
            if dz > 0:
                s[..., :dz, :, :] = 0
                if m is not None:
                    m[..., :dz, :, :] = 0
            elif dz < 0:
                s[..., dz:, :, :] = 0
                if m is not None:
                    m[..., dz:, :, :] = 0
            if dy > 0:
                s[..., :, :dy, :] = 0
                if m is not None:
                    m[..., :, :dy, :] = 0
            elif dy < 0:
                s[..., :, dy:, :] = 0
                if m is not None:
                    m[..., :, dy:, :] = 0
            if dx > 0:
                s[..., :, :, :dx] = 0
                if m is not None:
                    m[..., :, :, :dx] = 0
            elif dx < 0:
                s[..., :, :, dx:] = 0
                if m is not None:
                    m[..., :, :, dx:] = 0

        return s, m
