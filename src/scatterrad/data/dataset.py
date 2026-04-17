from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from scatterrad.config import PlansConfig, TaskConfig, TargetScope, TargetsSchema, load_case_targets
from scatterrad.models.scatter.scatter_cache import (
    load_cached_scatter,
    parse_scatter_cache_filename,
    scatter_cache_path,
    scatter_cache_paths,
)
from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.crop import read_crop


class ScatterRadDataset(Dataset):
    """Cache-backed dataset for scatter training in per-label or per-case scope."""

    def __init__(
        self,
        paths: ScatterRadPaths,
        basenames: list[str],
        schema: TargetsSchema,
        task: TaskConfig,
        plans: PlansConfig,
        labels: tuple[int, ...],
        augment: bool = False,
        scatter_cache_tensors: dict[tuple[str, int], torch.Tensor] | None = None,
        scatter_mask_tensors: dict[tuple[str, int], torch.Tensor] | None = None,
        cache_aug_variants: int = 0,
    ):
        self.paths = paths
        self.schema = schema
        self.task = task
        self.plans = plans
        self.labels = labels
        self.augment = augment
        self.cache_aug_variants = max(0, int(cache_aug_variants))
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

        self._validate_cache()

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
                    if scatter_cache_path(paths, basename, label_id).exists():
                        self.samples.append((basename, label_id))
        else:
            for basename in basenames:
                ct = self.case_targets.get(basename)
                if ct is None:
                    continue
                target = ct.get_per_case(task.target)
                if math.isnan(target):
                    continue
                has_any = any(
                    scatter_cache_path(paths, basename, label_id).exists() for label_id in labels
                )
                if has_any:
                    self.samples.append(basename)

    def __len__(self) -> int:
        return len(self.samples)

    def _validate_cache(self) -> None:
        missing: list[str] = []
        stale: list[str] = []
        for basename in self.case_targets:
            for label_id in self.labels:
                crop_path = self.paths.crop_path(basename, label_id)
                crop_exists = crop_path.exists()
                cache_paths = scatter_cache_paths(
                    self.paths,
                    basename,
                    label_id,
                    num_augmented_variants=self.cache_aug_variants,
                )
                if crop_exists:
                    for cache_path in cache_paths:
                        if not cache_path.exists():
                            missing.append(cache_path.stem)
                else:
                    for cache_path in cache_paths:
                        if cache_path.exists():
                            stale.append(cache_path.stem)
        for cache_path in self.paths.preprocessed_dataset_dir.joinpath("scatter_cache").glob("*.npy"):
            parsed = parse_scatter_cache_filename(cache_path)
            if parsed is None:
                continue
            basename, label_id, _variant_idx = parsed
            if not self.paths.crop_path(basename, label_id).exists():
                stale.append(cache_path.stem)
        if missing:
            preview = ", ".join(missing[:8])
            raise RuntimeError(
                "Scatter cache is incomplete for the current crops. "
                f"Missing cache entries for: {preview}"
            )
        if stale:
            preview = ", ".join(stale[:8])
            raise RuntimeError(
                "Scatter cache contains entries without matching crops. "
                f"Stale cache entries include: {preview}"
            )

    def _target_tensor(self, value: float) -> torch.Tensor:
        if self.target_type.value == "classification":
            return torch.tensor(int(value), dtype=torch.long)
        return torch.tensor(float(value), dtype=torch.float32)

    def _load_mask_tensor(self, basename: str, label_id: int, device: torch.device | None) -> torch.Tensor:
        _, mask, _ = read_crop(self.paths.crop_path(basename, label_id))
        mask_tensor = torch.from_numpy(mask[None, ...].astype(np.float32))
        if device is not None:
            mask_tensor = mask_tensor.to(device=device)
        return mask_tensor

    def _sample_variant_idx(self) -> int:
        if self.cache_aug_variants <= 0:
            return 0
        return int(self._rng.integers(0, self.cache_aug_variants + 1))

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if self.scope is TargetScope.PER_LABEL:
            basename, label_id = sample
            variant_idx = self._sample_variant_idx()
            key = (basename, int(label_id))
            scatter_tensor = self.scatter_cache_tensors.get(key)
            if scatter_tensor is None:
                scatter = load_cached_scatter(self.paths, basename, label_id, variant_idx=variant_idx)
                if scatter is None:
                    raise RuntimeError(
                        f"Scatter cache miss for {basename} label {label_id} variant {variant_idx}"
                    )
                scatter_tensor = torch.from_numpy(scatter)
            mask_tensor = None
            if self._need_cache_mask:
                mask_tensor = self.scatter_mask_tensors.get(key)
                if mask_tensor is None:
                    mask_tensor = self._load_mask_tensor(basename, label_id, scatter_tensor.device)
            if self.augment:
                aug_params = self._sample_aug_params()
                scatter_tensor, mask_tensor = self._apply_aug_params(scatter_tensor, mask_tensor, aug_params)
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

        basename = sample
        scatters = []
        masks = []
        present = []
        aug_params = self._sample_aug_params() if self.augment else None
        for label_id in self.labels:
            variant_idx = self._sample_variant_idx()
            key = (basename, int(label_id))
            if not scatter_cache_path(self.paths, basename, label_id).exists():
                scatters.append(None)
                if self._need_cache_mask:
                    masks.append(None)
                present.append(False)
                continue

            scatter_tensor = self.scatter_cache_tensors.get(key)
            if scatter_tensor is None:
                scatter = load_cached_scatter(self.paths, basename, label_id, variant_idx=variant_idx)
                if scatter is None:
                    raise RuntimeError(
                        f"Scatter cache miss for {basename} label {label_id} variant {variant_idx}"
                    )
                scatter_tensor = torch.from_numpy(scatter)

            mask_tensor = None
            if self._need_cache_mask:
                mask_tensor = self.scatter_mask_tensors.get(key)
                if mask_tensor is None:
                    mask_tensor = self._load_mask_tensor(basename, label_id, scatter_tensor.device)

            if aug_params is not None:
                scatter_tensor, mask_tensor = self._apply_aug_params(scatter_tensor, mask_tensor, aug_params)

            scatters.append(scatter_tensor)
            if self._need_cache_mask:
                masks.append(mask_tensor)
            present.append(True)

        target_value = self.case_targets[basename].get_per_case(self.task.target)

        ref = next((s for s in scatters if s is not None), None)
        if ref is None:
            raise RuntimeError(f"No scatter cache entries found for {basename}")
        filled = [s if s is not None else torch.zeros_like(ref) for s in scatters]
        item = {
            "scatter": torch.stack(filled, dim=0).float(),
            "target": self._target_tensor(target_value),
            "present": torch.tensor(present, dtype=torch.bool),
            "meta": {"basename": basename, "label_ids": list(self.labels)},
        }
        if self._need_cache_mask:
            ref_mask = next((m for m in masks if m is not None), None)
            if ref_mask is None:
                raise RuntimeError(f"No mask tensors found for {basename}")
            filled_masks = [m if m is not None else torch.zeros_like(ref_mask) for m in masks]
            item["masks"] = torch.stack(filled_masks, dim=0).float()
        return item

    def _sample_aug_params(self) -> dict:
        """Sample augmentation parameters once; apply consistently across all labels."""
        do_shift = self._rng.random() < 0.5
        shifts = [int(v) for v in self._rng.integers(-2, 3, size=3)] if do_shift else None
        return {"shifts": shifts}

    def _apply_aug_params(
        self,
        scatter: torch.Tensor,
        mask: torch.Tensor | None,
        params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply pre-sampled augmentation parameters to a (C,D,H,W) scatter tensor."""
        s = scatter
        m = mask

        if params["shifts"] is not None:
            shifts = params["shifts"]
            s = torch.roll(s, shifts=tuple(shifts), dims=(-3, -2, -1))
            if m is not None:
                m = torch.roll(m, shifts=tuple(shifts), dims=(-3, -2, -1))
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
