from __future__ import annotations

import torch
from torch import nn

from scatterrad.config import TargetScope, TargetType
from scatterrad.models.scatter.backend import ScatterBackend
from scatterrad.models.scatter.frontend import WaveletFrontend
from scatterrad.models.scatter.pooling import MaskedAttentionPool


class ScatterRadModel(nn.Module):
    """Cache-backed scatter model for per-label and per-case tasks."""

    def __init__(
        self,
        crop_size: tuple[int, int, int],
        target_type: TargetType,
        target_scope: TargetScope,
        num_classes: int | None,
        spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        wavelet: str = "coif1",
        level: int = 1,
        log_sigmas_mm: tuple[float, ...] = (1.0, 2.0, 3.0),
        use_gradient: bool = True,
        # Legacy alias: J= maps to level=
        J: int | None = None,
        L: int | None = None,
        hidden_channels: int = 32,
        dropout: float = 0.3,
        mask_mode: str = "zero",
        second_order: bool = True,
        scatter_out_channels: int | None = None,
        scatter_out_shape: tuple | None = None,
    ):
        super().__init__()
        self.target_type = target_type
        self.target_scope = target_scope

        effective_level = int(J) if J is not None else level

        self.frontend = WaveletFrontend(
            crop_size=crop_size,
            spacing_mm=spacing_mm,
            wavelet=wavelet,
            level=effective_level,
            log_sigmas_mm=log_sigmas_mm,
            use_gradient=use_gradient,
            mask_mode=mask_mode,
            out_channels_override=scatter_out_channels,
            out_shape_override=scatter_out_shape,
        )
        self.backend = ScatterBackend(
            in_channels=self.frontend.out_channels,
            hidden_channels=hidden_channels,
            in_shape=self.frontend.out_shape,
            second_order=second_order,
        )
        self.pool = MaskedAttentionPool(hidden_channels)

        out_dim = (
            1
            if (target_type is TargetType.REGRESSION or (num_classes or 2) == 2)
            else int(num_classes)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_dim),
        )

    def _encode_from_scatter(
        self, scatter: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode pre-computed filter-bank channels, bypassing the frontend."""
        if mask is None:
            mask = torch.ones(scatter.shape[0], 1, *scatter.shape[2:], device=scatter.device)
        return self.backend(scatter, mask)

    def forward(self, batch: dict) -> dict:
        if "scatter" not in batch:
            raise ValueError("ScatterRadModel expects precomputed scatter tensors in batch['scatter']")

        if self.target_scope is TargetScope.PER_LABEL:
            feats = self._encode_from_scatter(batch["scatter"], batch.get("mask"))
            logits = self.head(feats)
            return {"logits": logits}

        present = batch["present"]
        scatter = batch["scatter"]  # (B, N, C, D, H, W)
        b, n = scatter.shape[:2]
        masks = batch.get("masks")
        if masks is not None:
            feats = self._encode_from_scatter(
                scatter.view(b * n, *scatter.shape[2:]),
                masks.view(b * n, *masks.shape[2:]),
            )
        else:
            feats = self._encode_from_scatter(scatter.view(b * n, *scatter.shape[2:]))
        feats = feats.view(b, n, -1)
        pooled, attn = self.pool(feats, present)
        logits = self.head(pooled)
        return {"logits": logits, "attention_weights": attn}
