from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import amp
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from scatterrad.config import (
    DatasetConfig,
    ModelKind,
    PlansConfig,
    TaskConfig,
    TargetScope,
    TargetType,
    TargetsSchema,
)
from scatterrad.data import ClassBalancedSampler, ScatterRadDataset, scatter_collate_fn
from scatterrad.evaluation import compute_metrics
from scatterrad.models.scatter.model import ScatterRadModel
from scatterrad.models.scatter.scatter_cache import load_cached_scatter, precompute_and_cache
from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.crop import read_crop
from scatterrad.utils import resolve_num_workers

log = logging.getLogger(__name__)


def _load_splits(path: Path) -> list[dict[str, list[str]]]:
    return json.loads(path.read_text())["folds"]


def _loss_fn(spec_type: TargetType, num_classes: int | None, pos_weight: torch.Tensor | None = None) -> nn.Module:
    if spec_type is TargetType.REGRESSION:
        return nn.SmoothL1Loss()
    if (num_classes or 2) == 2:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.CrossEntropyLoss()


def _predict_from_logits(logits: torch.Tensor, spec_type: TargetType, num_classes: int | None):
    if spec_type is TargetType.REGRESSION:
        return logits.squeeze(-1), None
    if (num_classes or 2) == 2:
        probs = torch.sigmoid(logits.squeeze(-1))
        pred = (probs >= 0.5).long()
        return pred, torch.stack([1 - probs, probs], dim=1)
    probs = torch.softmax(logits, dim=1)
    return probs.argmax(dim=1), probs


def _loader_num_workers(task: TaskConfig, dataset_size: int) -> int:
    requested = task.model_config.get("num_workers")
    return resolve_num_workers(requested if requested is not None else 0, max_tasks=dataset_size)


def _load_state_dict_from_checkpoint(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported checkpoint format: expected dict or {'state_dict': ...}")


def _plot_training_log(df: pd.DataFrame, out_path: Path, target_type: TargetType) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = df["epoch"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, df["train_loss"], label="train")
    if "val_loss" in df.columns:
        axes[0].plot(epochs, df["val_loss"], label="val")
    if "is_best" in df.columns:
        best_epochs = epochs[df["is_best"].to_numpy().astype(bool)]
        for be in best_epochs:
            axes[0].axvline(be, color="green", alpha=0.3, linewidth=1)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Val metric
    if target_type is TargetType.REGRESSION:
        metric_col = "mae"
    else:
        metric_col = "auc" if "auc" in df.columns else "auc_macro"
    if metric_col in df.columns:
        axes[1].plot(epochs, df[metric_col], color="tab:orange", label=metric_col)
        axes[1].set_title(f"Val {metric_col}")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # LR
    if "lr" in df.columns:
        axes[2].plot(epochs, df["lr"], color="tab:red")
        axes[2].set_title("Learning rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _select_slice_index(mask_3d: np.ndarray) -> int:
    fg = np.argwhere(mask_3d > 0.5)
    if fg.size == 0:
        return int(mask_3d.shape[0] // 2)
    return int(np.median(fg[:, 0]))


def _norm01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _robust_window01(
    arr: np.ndarray,
    mask: np.ndarray | None = None,
    q_low: float = 1.0,
    q_high: float = 99.0,
) -> tuple[np.ndarray, float, float]:
    arr = arr.astype(np.float32, copy=False)
    vals = arr
    if mask is not None:
        fg = mask > 0
        if np.any(fg):
            vals = arr[fg]
    vals = np.asarray(vals, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32), 0.0, 1.0
    lo = float(np.nanpercentile(vals, q_low))
    hi = float(np.nanpercentile(vals, q_high))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi - lo < 1e-8):
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi - lo < 1e-8):
        return np.zeros_like(arr, dtype=np.float32), lo, hi
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32, copy=False), lo, hi


def _save_debug_panels(
    *,
    model: ScatterRadModel,
    dataset: ScatterRadDataset,
    result_dir: Path,
    paths: ScatterRadPaths,
    device: torch.device,
    epoch: int,
    target_type: TargetType,
    num_classes: int | None,
    num_cases: int,
    split_name: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("Debug panels disabled (matplotlib unavailable): %s", exc)
        return

    if len(dataset) == 0:
        return

    debug_dir = result_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    n_cases = min(int(num_cases), len(dataset))
    if n_cases <= 0:
        return
    if n_cases >= len(dataset):
        sample_indices = list(range(len(dataset)))
    else:
        # Spread debug picks across the split instead of always taking the first
        # few entries (which can over-represent one case/label pattern).
        sample_indices = np.linspace(0, len(dataset) - 1, n_cases, dtype=int).tolist()
        sample_indices = list(dict.fromkeys(int(i) for i in sample_indices))
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            meta = sample.get("meta", {})
            basename = str(meta.get("basename", f"case{idx:03d}"))
            label_id = meta.get("label_id", None)
            label_suffix = f"_label{int(label_id):03d}" if label_id is not None else ""

            if "scatter" in sample:
                scatter = sample["scatter"]
                if scatter.ndim == 5:
                    present = sample.get("present")
                    if present is not None and torch.is_tensor(present):
                        present_idx = torch.where(present)[0]
                        pick = int(present_idx[0]) if len(present_idx) else 0
                    else:
                        pick = 0
                    scatter = scatter[pick]
                scatter_t = scatter.detach().to(device=device, dtype=torch.float32).unsqueeze(0)
            elif "image" in sample and "mask" in sample:
                image_t = sample["image"].detach().to(device=device, dtype=torch.float32).unsqueeze(0)
                mask_t = sample["mask"].detach().to(device=device, dtype=torch.float32).unsqueeze(0)
                scatter_t, _ = model.frontend(image_t, mask_t)
            else:
                continue

            mask_np: np.ndarray
            image_np: np.ndarray
            if label_id is not None:
                crop_path = paths.crop_path(basename, int(label_id))
                if crop_path.exists():
                    image_np, mask_np, _ = read_crop(crop_path)
                else:
                    mask_np = np.ones(scatter_t.shape[-3:], dtype=np.uint8)
                    image_np = scatter_t[0, 0].detach().cpu().numpy().astype(np.float32)
            else:
                mask_np = np.ones(scatter_t.shape[-3:], dtype=np.uint8)
                image_np = scatter_t[0, 0].detach().cpu().numpy().astype(np.float32)

            with torch.no_grad():
                conv_feat = model.backend.conv(scatter_t)
                attn_map = conv_feat.abs().mean(dim=1)[0].detach().cpu().numpy().astype(np.float32)

                model_batch: dict[str, Any] = {}
                for k, v in sample.items():
                    if k == "meta":
                        continue
                    if torch.is_tensor(v):
                        model_batch[k] = v.unsqueeze(0).to(device)
                    else:
                        model_batch[k] = v
                if "scatter" in model_batch and torch.is_tensor(model_batch["scatter"]):
                    model_batch["scatter"] = model_batch["scatter"].to(dtype=torch.float32)
                out = model(model_batch)
                logits = out["logits"]

            if target_type is TargetType.REGRESSION:
                pred_text = f"pred={float(logits.squeeze().detach().cpu().item()):.3f}"
            elif (num_classes or 2) == 2:
                prob = float(torch.sigmoid(logits.squeeze()).detach().cpu().item())
                pred_text = f"p1={prob:.3f}"
            else:
                probs = torch.softmax(logits[0], dim=0).detach().cpu().numpy()
                pred_cls = int(np.argmax(probs))
                pred_text = f"pred={pred_cls} p={float(np.max(probs)):.3f}"

            target_val = sample.get("target")
            if torch.is_tensor(target_val):
                target_text = f"target={float(target_val.detach().cpu().item()):.3f}"
            else:
                target_text = "target=n/a"

            scatter_np = scatter_t[0].detach().cpu().numpy().astype(np.float32)
            z = _select_slice_index(mask_np)
            img_disp, img_lo, img_hi = _robust_window01(image_np, mask=mask_np)
            img2d = img_disp[z]
            msk2d = mask_np[z]
            attn_disp, _attn_lo, _attn_hi = _robust_window01(attn_map)
            attn2d = attn_disp[z]

            n_channels = int(scatter_np.shape[0])
            n_panels = 3 + n_channels
            n_cols = 4
            n_rows = int(math.ceil(n_panels / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
            axes = np.asarray(axes).reshape(-1)

            axes[0].imshow(img2d, cmap="gray")
            axes[0].set_title(f"Input image p1-p99 [{img_lo:.2f}, {img_hi:.2f}]")
            axes[0].axis("off")

            axes[1].imshow(img2d, cmap="gray")
            axes[1].imshow(msk2d, cmap="Reds", alpha=0.35)
            axes[1].set_title("Mask overlay")
            axes[1].axis("off")

            axes[2].imshow(img2d, cmap="gray")
            axes[2].imshow(attn2d, cmap="jet", alpha=np.clip(attn2d, 0, 1) * 0.75, vmin=0, vmax=1)
            axes[2].set_title("Network attention")
            axes[2].axis("off")

            for ch in range(n_channels):
                ax = axes[3 + ch]
                ch_disp, _, _ = _robust_window01(scatter_np[ch], mask=mask_np)
                ax.imshow(ch_disp[z], cmap="gray")
                ax.set_title(f"feature ch{ch:02d}")
                ax.axis("off")

            for k in range(3 + n_channels, len(axes)):
                axes[k].axis("off")

            fig.suptitle(
                f"epoch {epoch:03d}  split={split_name}  case {basename}{label_suffix}  {target_text}  {pred_text}",
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(
                debug_dir / f"epoch{epoch:03d}_{split_name}_{basename}{label_suffix}.png",
                dpi=100,
            )
            plt.close(fig)
        except Exception as exc:
            log.warning("Debug panel failed for index %d: %s", idx, exc)


def train(
    paths: ScatterRadPaths,
    task: TaskConfig,
    dataset: DatasetConfig,
    schema: TargetsSchema,
    plans: PlansConfig,
    fold: int | None = None,
    continue_existing: bool = False,
    resume_from: Path | None = None,
) -> None:
    """Train scatter model with cross-validation."""

    if task.model is not ModelKind.SCATTER:
        raise ValueError("Scatter trainer called for non-scatter task")

    spec = schema[task.target]
    labels = task.resolved_labels(schema)
    splits_path = paths.results_splits_json if paths.results_splits_json.exists() else paths.splits_json
    folds = _load_splits(splits_path)
    selected = range(len(folds)) if fold is None else [int(fold)]

    _ = dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "Task: %s  target: %s  model: scatter  device: %s  folds: %s",
        task.name, task.target, device, len(list(selected)),
    )
    selected = range(len(folds)) if fold is None else [int(fold)]

    def _gpu_preload_budget_bytes(device_obj: torch.device) -> int:
        if device_obj.type != "cuda":
            return 0
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device_obj)
        except Exception:
            return 0
        frac = float(
            task.model_config.get(
                "gpu_preload_fraction",
                os.environ.get("SCATTERRAD_GPU_PRELOAD_FRACTION", "0.7"),
            )
        )
        frac = min(max(frac, 0.0), 1.0)
        reserve_gb = float(
            task.model_config.get(
                "gpu_preload_reserve_gb",
                os.environ.get("SCATTERRAD_GPU_PRELOAD_RESERVE_GB", "4"),
            )
        )
        max_gb = float(
            task.model_config.get(
                "gpu_preload_max_gb",
                os.environ.get("SCATTERRAD_GPU_PRELOAD_MAX_GB", "0"),
            )
        )
        budget = int(float(free_bytes) * frac) - int(reserve_gb * (1024**3))
        if max_gb > 0:
            budget = min(budget, int(max_gb * (1024**3)))
        return max(0, budget)

    def _preload_dataset_scatter_to_gpu(
        ds: ScatterRadDataset,
        device_obj: torch.device,
        dtype: torch.dtype,
        max_bytes: int,
    ) -> tuple[dict[tuple[str, int], torch.Tensor], dict[tuple[str, int], torch.Tensor], int]:
        keys: set[tuple[str, int]] = set()
        if ds.scope is TargetScope.PER_LABEL:
            for item in ds.samples:
                basename, label_id = item
                keys.add((basename, int(label_id)))
        else:
            for basename in ds.samples:
                for label_id in ds.labels:
                    keys.add((basename, int(label_id)))

        scatter_tensors: dict[tuple[str, int], torch.Tensor] = {}
        mask_tensors: dict[tuple[str, int], torch.Tensor] = {}
        need_mask = bool(task.model_config.get("second_order", False)) and ds.scope is TargetScope.PER_LABEL

        total_bytes = 0
        for basename, label_id in sorted(keys):
            scatter = load_cached_scatter(paths, basename, label_id)
            if scatter is None:
                continue
            bytes_scatter = scatter.size * torch.tensor([], dtype=dtype).element_size()
            if max_bytes > 0 and (total_bytes + bytes_scatter) > max_bytes:
                continue
            t = torch.from_numpy(scatter).to(device=device_obj, dtype=dtype, non_blocking=True)
            scatter_tensors[(basename, label_id)] = t
            total_bytes += t.numel() * t.element_size()

            if need_mask:
                _, mask, _ = read_crop(paths.crop_path(basename, label_id))
                bytes_mask = int(mask[None, ...].size * 4)
                if max_bytes > 0 and (total_bytes + bytes_mask) > max_bytes:
                    continue
                mt = torch.from_numpy(mask[None, ...].astype(np.float32)).to(
                    device=device_obj, non_blocking=True
                )
                mask_tensors[(basename, label_id)] = mt
                total_bytes += mt.numel() * mt.element_size()

        log.info(
            "Preloaded %d scatter cache tensors to %s (%.2f GB, dtype=%s)",
            len(scatter_tensors),
            device_obj,
            total_bytes / (1024**3),
            str(dtype).replace("torch.", ""),
        )
        return scatter_tensors, mask_tensors, total_bytes

    for fold_idx in selected:
        result_dir = paths.result_dir(
            task.name,
            "scatter",
            fold_idx,
            planner_name=plans.planner,
            trainer_name="scatterradDefaultTrainer",
        )
        fold_checkpoint = result_dir / "checkpoint.pt"
        fold_metrics = result_dir / "metrics.json"
        if continue_existing and fold_metrics.exists():
            log.info("Fold %d/%d: skipping (already done)", fold_idx, len(folds) - 1)
            continue

        fold_info = folds[fold_idx]
        cache_dir = paths.preprocessed_dataset_dir / "scatter_cache"
        has_cache = cache_dir.exists() and any(cache_dir.glob("*.npy"))
        if has_cache:
            log.info("Fold %d/%d: scatter cache found — skipping live frontend computation", fold_idx, len(folds) - 1)
        train_ds = ScatterRadDataset(
            paths=paths,
            basenames=fold_info["train"],
            schema=schema,
            task=task,
            plans=plans,
            labels=labels,
            augment=bool(task.model_config.get("augment", True)),
            use_scatter_cache=has_cache,
        )
        val_ds = ScatterRadDataset(
            paths=paths,
            basenames=fold_info["val"],
            schema=schema,
            task=task,
            plans=plans,
            labels=labels,
            augment=False,
            use_scatter_cache=has_cache,
        )

        preload_scatter_to_gpu = bool(task.model_config.get("preload_scatter_to_gpu", False))
        if not preload_scatter_to_gpu:
            preload_scatter_to_gpu = os.environ.get("SCATTERRAD_PRELOAD_SCATTER_TO_GPU", "").strip() in {
                "1",
                "true",
                "TRUE",
                "yes",
                "YES",
            }
        if preload_scatter_to_gpu and has_cache and device.type == "cuda":
            gpu_dtype_name = str(
                task.model_config.get(
                    "gpu_scatter_dtype",
                    os.environ.get("SCATTERRAD_GPU_SCATTER_DTYPE", "float16"),
                )
            ).lower()
            gpu_dtype = torch.float16 if gpu_dtype_name == "float16" else torch.float32
            total_budget = _gpu_preload_budget_bytes(device)
            train_share = float(
                task.model_config.get(
                    "gpu_preload_train_share",
                    os.environ.get("SCATTERRAD_GPU_PRELOAD_TRAIN_SHARE", "0.8"),
                )
            )
            train_share = min(max(train_share, 0.0), 1.0)
            train_budget = int(total_budget * train_share)
            val_budget = max(0, total_budget - train_budget)
            train_scatter, train_masks, used_train = _preload_dataset_scatter_to_gpu(
                train_ds, device, gpu_dtype, max_bytes=train_budget
            )
            remaining_for_val = max(0, val_budget - max(0, used_train - train_budget))
            val_scatter, val_masks, _used_val = _preload_dataset_scatter_to_gpu(
                val_ds, device, gpu_dtype, max_bytes=remaining_for_val
            )
            train_ds.scatter_cache_tensors = train_scatter
            train_ds.scatter_mask_tensors = train_masks
            val_ds.scatter_cache_tensors = val_scatter
            val_ds.scatter_mask_tensors = val_masks
            log.info("GPU scatter preload enabled; forcing dataloader workers=0 for CUDA safety")

        if len(train_ds) == 0 or len(val_ds) == 0:
            log.warning("Fold %d/%d: skipping due to empty dataset", fold_idx, len(folds) - 1)
            continue

        log.info(
            "Fold %d/%d: %d train samples, %d val samples",
            fold_idx, len(folds) - 1, len(train_ds), len(val_ds),
        )

        # If scatter cache exists, read out_channels/out_shape from a cache entry
        # to avoid running the expensive kymatio dummy forward during model init.
        scatter_out_channels: int | None = None
        scatter_out_shape: tuple | None = None
        if has_cache:
            sample_npy = next(cache_dir.glob("*.npy"), None)
            if sample_npy is not None:
                arr = np.load(sample_npy)  # shape: (C, D, H, W)
                scatter_out_channels = int(arr.shape[0])
                scatter_out_shape = tuple(int(x) for x in arr.shape[1:])
                log.info(
                    "Fold %d/%d: scatter cache — out_channels=%d  out_shape=%s",
                    fold_idx, len(folds) - 1, scatter_out_channels, scatter_out_shape,
                )

        wavelet = str(task.model_config.get("wavelet", "coif1"))
        level = int(task.model_config.get("level", task.model_config.get("J", 1)))
        log_sigmas_mm = tuple(float(s) for s in task.model_config.get("log_sigmas_mm", [1.0, 2.0, 3.0]))
        use_gradient = bool(task.model_config.get("use_gradient", True))
        second_order = bool(task.model_config.get("second_order", False))
        spacing_mm = tuple(float(s) for s in plans.target_spacing_mm)
        log.info(
            "Fold %d/%d: initializing filter-bank model  wavelet=%s level=%d  "
            "log_sigmas=%s  gradient=%s  second_order=%s",
            fold_idx, len(folds) - 1, wavelet, level,
            log_sigmas_mm, use_gradient, second_order,
        )
        model = ScatterRadModel(
            crop_size=plans.crop_size_voxels,
            target_type=spec.type,
            target_scope=spec.scope,
            num_classes=spec.num_classes,
            spacing_mm=spacing_mm,
            wavelet=wavelet,
            level=level,
            log_sigmas_mm=log_sigmas_mm,
            use_gradient=use_gradient,
            hidden_channels=int(task.model_config.get("conv_channels", 24)),
            dropout=float(task.model_config.get("dropout", 0.4)),
            mask_mode=str(task.model_config.get("mask_mode", "zero")),
            second_order=second_order,
            scatter_out_channels=scatter_out_channels,
            scatter_out_shape=scatter_out_shape,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            "Fold %d/%d: model ready  frontend_channels=%d  out_shape=%s  backend=cnn  params=%d",
            fold_idx, len(folds) - 1,
            model.frontend.out_channels, model.frontend.out_shape, n_params,
        )

        if bool(task.model_config.get("cache_scatter_output", False)):
            precompute_and_cache(paths, model.frontend, device=str(device))

        sampler = ClassBalancedSampler(train_ds, seed=task.cv.seed)
        train_workers = _loader_num_workers(task, len(train_ds))
        val_workers = _loader_num_workers(task, len(val_ds))
        if preload_scatter_to_gpu and has_cache and device.type == "cuda":
            train_workers = 0
            val_workers = 0
        use_cuda = device.type == "cuda" and not (
            preload_scatter_to_gpu and has_cache and device.type == "cuda"
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(task.model_config.get("batch_size", 16)),
            sampler=sampler,
            collate_fn=scatter_collate_fn,
            num_workers=train_workers,
            pin_memory=use_cuda,
            persistent_workers=train_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(task.model_config.get("batch_size", 16)),
            shuffle=False,
            collate_fn=scatter_collate_fn,
            num_workers=val_workers,
            pin_memory=use_cuda,
            persistent_workers=val_workers > 0,
        )

        max_epochs = int(task.model_config.get("epochs", 100))

        # Compute pos_weight for binary classification from training labels
        pos_weight: torch.Tensor | None = None
        if spec.type is TargetType.CLASSIFICATION and (spec.num_classes or 2) == 2:
            train_targets = np.asarray([
                int(train_ds.case_targets[
                    s[0] if isinstance(s, tuple) else s
                ].get_per_label(task.target, s[1]) if isinstance(s, tuple)
                else train_ds.case_targets[s].get_per_case(task.target))
                for s in train_ds.samples
            ])
            n_pos = int(train_targets.sum())
            n_neg = len(train_targets) - n_pos
            if n_pos > 0 and n_neg > 0:
                # Use sqrt of ratio to soften the imbalance correction.
                # Raw n_neg/n_pos drives the model to predict all-positive
                # even when a balanced sampler is already used.
                pos_weight = torch.tensor([np.sqrt(n_neg / n_pos)], dtype=torch.float32).to(device)
                log.info("Fold %d/%d: pos_weight=%.2f  (n_pos=%d n_neg=%d)",
                    fold_idx, len(folds) - 1, float(pos_weight), n_pos, n_neg)

        # AdamW + cosine annealing — more stable than SGD for the small scatter
        # feature space (model sees O(100) values per sample, not full image patches).
        lr = float(task.model_config.get("lr", 3e-5))
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(task.model_config.get("weight_decay", 5e-4)),
        )
        eta_min = float(task.model_config.get("lr_min", 1e-6))
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)
        loss_fn = _loss_fn(spec.type, spec.num_classes, pos_weight=pos_weight)
        amp_enabled = bool(
            task.model_config.get(
                "amp",
                os.environ.get("SCATTERRAD_AMP", "1") in {"1", "true", "TRUE", "yes", "YES"},
            )
        ) and (device.type == "cuda")
        scaler = amp.GradScaler("cuda", enabled=amp_enabled)
        start_epoch = 0
        best_metric = -float("inf")
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        early_stopping_patience = int(task.model_config.get("early_stopping_patience", 12))
        early_stopping_min_epochs = int(task.model_config.get("early_stopping_min_epochs", 20))

        resume_path: Path | None = None
        if continue_existing and fold_checkpoint.exists():
            resume_path = fold_checkpoint
        elif resume_from is not None:
            resume_path = resume_from
        if resume_path is not None:
            payload = torch.load(resume_path, map_location=device)
            state_dict = _load_state_dict_from_checkpoint(payload)
            model.load_state_dict(state_dict, strict=False)
            log.info("Loaded model weights from %s", resume_path)
            if resume_path == fold_checkpoint and continue_existing and isinstance(payload, dict):
                if "optimizer" in payload and isinstance(payload["optimizer"], dict):
                    optimizer.load_state_dict(payload["optimizer"])
                if "epoch" in payload:
                    try:
                        start_epoch = int(payload["epoch"]) + 1
                    except (TypeError, ValueError):
                        start_epoch = 0
                if "best_metric" in payload:
                    try:
                        best_metric = float(payload["best_metric"])
                    except (TypeError, ValueError):
                        best_metric = -float("inf")

        log.info(
            "Fold %d/%d: training for %d epochs  lr=%.0e  eta_min=%.0e  batch=%d  "
            "opt=AdamW  lr_schedule=cosine  backend=cnn  params=%d",
            fold_idx, len(folds) - 1, max_epochs, lr, eta_min,
            int(task.model_config.get("batch_size", 16)),
            sum(p.numel() for p in model.parameters()),
        )
        debug_enabled = bool(task.model_config.get("debug", False))
        debug_every = max(1, int(task.model_config.get("debug_save_every_n_epochs", 10)))
        debug_cases_default = max(1, int(task.model_config.get("debug_num_cases", 2)))
        debug_cases_train = max(
            1,
            int(task.model_config.get("debug_num_cases_train", debug_cases_default)),
        )
        debug_cases_val = max(
            1,
            int(task.model_config.get("debug_num_cases_val", debug_cases_default)),
        )

        training_log: list[dict] = []
        t0 = perf_counter()
        for epoch in range(start_epoch, max_epochs):
            sampler.set_epoch(epoch)
            model.train()
            epoch_losses = []
            for batch in train_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                if (
                    "scatter" in batch
                    and torch.is_tensor(batch["scatter"])
                    and not amp_enabled
                ):
                    batch["scatter"] = batch["scatter"].float()
                with amp.autocast("cuda", enabled=amp_enabled):
                    out = model(batch)
                    logits = out["logits"]
                    target = batch["target"]
                    if spec.type is TargetType.REGRESSION:
                        loss = loss_fn(logits.squeeze(-1), target.float())
                    elif (spec.num_classes or 2) == 2:
                        loss = loss_fn(logits.squeeze(-1), target.float())
                    else:
                        loss = loss_fn(logits, target.long())
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(float(loss.detach()))

            model.eval()
            y_true = []
            y_pred = []
            y_proba = []
            attn_values = []
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {
                        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
                    }
                    if (
                        "scatter" in batch
                        and torch.is_tensor(batch["scatter"])
                        and not amp_enabled
                    ):
                        batch["scatter"] = batch["scatter"].float()
                    with amp.autocast("cuda", enabled=amp_enabled):
                        out = model(batch)
                        logits = out["logits"]
                        target = batch["target"]
                        if spec.type is TargetType.REGRESSION:
                            val_loss = loss_fn(logits.squeeze(-1), target.float())
                        elif (spec.num_classes or 2) == 2:
                            val_loss = loss_fn(logits.squeeze(-1), target.float())
                        else:
                            val_loss = loss_fn(logits, target.long())
                    val_losses.append(float(val_loss))
                    pred, proba = _predict_from_logits(logits, spec.type, spec.num_classes)
                    y_true.extend(target.cpu().numpy().tolist())
                    y_pred.extend(pred.cpu().numpy().tolist())
                    if proba is not None:
                        y_proba.append(proba.cpu().numpy())
                    if "attention_weights" in out:
                        attn_values.append(out["attention_weights"].cpu().numpy())

            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            y_proba_arr = np.concatenate(y_proba, axis=0) if y_proba else None
            metrics = compute_metrics(
                y_true_arr, y_pred_arr, y_proba_arr, spec.type, spec.num_classes
            )

            if spec.type is TargetType.REGRESSION:
                score = -float(metrics["mae"])
                score_str = f"mae={-score:.4f}"
            else:
                score = float(metrics.get("auc", metrics.get("auc_macro", 0.0)) or 0.0)
                if np.isnan(score):
                    score = 0.0
                score_str = f"auc={score:.4f}"

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            is_best = score > best_metric
            log.info(
                "Fold %d/%d  epoch %d/%d  loss=%.4f  val_loss=%.4f  %s  lr=%.2e%s",
                fold_idx, len(folds) - 1,
                epoch + 1, max_epochs,
                float(np.mean(epoch_losses)),
                float(np.mean(val_losses)),
                score_str,
                current_lr,
                " *" if is_best else "",
            )

            log_row = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(epoch_losses)),
                "val_loss": float(np.mean(val_losses)),
                "lr": current_lr,
                "is_best": is_best,
            }
            log_row.update({k: v for k, v in metrics.items() if isinstance(v, float)})
            training_log.append(log_row)

            result_dir.mkdir(parents=True, exist_ok=True)
            df_log = pd.DataFrame(training_log)
            df_log.to_csv(result_dir / "training_log.csv", index=False)
            _plot_training_log(df_log, result_dir / "training_log.png", spec.type)
            if debug_enabled and (((epoch + 1) % debug_every) == 0 or (epoch + 1) == max_epochs):
                _save_debug_panels(
                    model=model,
                    dataset=train_ds,
                    result_dir=result_dir,
                    paths=paths,
                    device=device,
                    epoch=epoch + 1,
                    target_type=spec.type,
                    num_classes=spec.num_classes,
                    num_cases=debug_cases_train,
                    split_name="train",
                )
                _save_debug_panels(
                    model=model,
                    dataset=val_ds,
                    result_dir=result_dir,
                    paths=paths,
                    device=device,
                    epoch=epoch + 1,
                    target_type=spec.type,
                    num_classes=spec.num_classes,
                    num_cases=debug_cases_val,
                    split_name="val",
                )
                log.info(
                    "Fold %d/%d  epoch %d: wrote debug panels to %s",
                    fold_idx,
                    len(folds) - 1,
                    epoch + 1,
                    result_dir / "debug",
                )

            if is_best:
                best_metric = score
                best_epoch = epoch
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                early_stopping_patience > 0
                and (epoch + 1) >= early_stopping_min_epochs
                and epochs_without_improvement >= early_stopping_patience
            ):
                log.info(
                    "Fold %d/%d: early stopping at epoch %d (no improvement for %d epochs, best epoch=%d)",
                    fold_idx,
                    len(folds) - 1,
                    epoch + 1,
                    epochs_without_improvement,
                    best_epoch + 1,
                )
                break

        runtime = perf_counter() - t0
        if best_state is not None:
            model.load_state_dict(best_state)

        result_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": best_epoch,
                "best_metric": float(best_metric),
                "task": task.name,
            },
            result_dir / "checkpoint.pt",
        )

        pred_df = pd.DataFrame({"y_true": y_true_arr, "y_pred": y_pred_arr})
        pred_df.to_csv(result_dir / "predictions.csv", index=False)

        payload = {
            "task": task.name,
            "model": "scatter",
            "fold": fold_idx,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "target_type": spec.type.value,
            "metrics": metrics,
            "runtime_seconds": float(runtime),
        }
        if attn_values:
            mean_attn = np.concatenate(attn_values, axis=0).mean(axis=0)
            payload["attention_weights_mean"] = {
                str(label): float(w) for label, w in zip(labels, mean_attn)
            }
        (result_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        (result_dir / "log.txt").write_text(
            f"best_epoch={best_epoch}\nbest_metric={best_metric:.6f}\n"
        )
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        log.info(
            "Fold %d/%d done in %.1fs  best_epoch=%d  →  %s",
            fold_idx, len(folds) - 1, runtime, best_epoch + 1, metrics_str,
        )
