from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scatterrad.models.scatter.frontend import _second_order_features, _N_GLCM_STATS


def masked_gap(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked global average pooling over spatial dims (2,3,4)."""
    denom = mask.sum(dim=(2, 3, 4)).clamp(min=1.0)
    return (feat * mask).sum(dim=(2, 3, 4)) / denom


def _batch_second_order(
    bands: torch.Tensor,  # (B, C, D, H, W)
    mask: torch.Tensor,   # (B, 1, D, H, W)
    n_bins: int = 32,
) -> torch.Tensor:
    """GLCM-derived second-order features for each sample. Returns (B, C*4)."""
    device = bands.device
    b, C = bands.shape[0], bands.shape[1]
    bands_np = bands.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()[:, 0]
    results = np.zeros((b, C * _N_GLCM_STATS), dtype=np.float32)
    for i in range(b):
        results[i] = _second_order_features(bands_np[i], mask_np[i], n_bins=n_bins)
    return torch.from_numpy(results).to(device=device, dtype=torch.float32)


class ScatterBackend(nn.Module):
    """Convolutional backend over the filter-bank feature maps.

    Architecture:
        Input: (B, C_in, D, H, W)  — 1 original image + filter-bank channels
               ↓
        Conv3d(C_in → hidden, k=3, pad=1) + BN + ReLU
               ↓
        MaxPool3d(k=2, s=2)
               ↓
        Conv3d(hidden → 2*hidden, k=3, s=2, pad=1) + BN + ReLU
               ↓
        MaxPool3d(k=2, s=2)
               ↓
        Masked global average pool (inside ROI mask) → (B, 2*hidden)
               ↓
        LayerNorm → Linear(hidden → hidden) → ReLU

    Optionally, GLCM second-order features can be concatenated after GAP
    for ablation studies (second_order=True).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        in_shape: tuple | None = None,  # kept for API compat, unused
        second_order: bool = False,
        glcm_bins: int = 32,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.second_order = second_order
        self.glcm_bins = glcm_bins
        self.conv_out_channels = hidden_channels * 2

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(hidden_channels, self.conv_out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(self.conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        glcm_dim = in_channels * _N_GLCM_STATS if second_order else 0
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.conv_out_channels + glcm_dim),
            nn.Linear(self.conv_out_channels + glcm_dim, hidden_channels),
            nn.ReLU(inplace=True),
        )

    def _downsample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = F.max_pool3d(mask, kernel_size=2, stride=2)
        mask = F.max_pool3d(mask, kernel_size=3, stride=2, padding=1)
        mask = F.max_pool3d(mask, kernel_size=2, stride=2)
        return mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        conv_feat = self.conv(x)
        pooled_mask = self._downsample_mask(mask).to(dtype=conv_feat.dtype)
        feat = masked_gap(conv_feat, pooled_mask)  # (B, 2*hidden)
        if self.second_order:
            glcm = _batch_second_order(x, mask, n_bins=self.glcm_bins)
            feat = torch.cat([feat, glcm], dim=1)
        return self.mlp(feat)
