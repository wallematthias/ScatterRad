from __future__ import annotations

import torch
from torch import nn


class MaskedAttentionPool(nn.Module):
    """Attention pooling over variable set of label embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(
        self, feats: torch.Tensor, present_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(feats).squeeze(-1)
        logits = logits.masked_fill(~present_mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        weights = weights * present_mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = weights / denom
        pooled = (feats * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights
