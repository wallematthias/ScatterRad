from __future__ import annotations

from typing import Any

import torch


def _pad_spatial_to(t: torch.Tensor, target_dhw: tuple[int, int, int]) -> torch.Tensor:
    d, h, w = t.shape[-3:]
    td, th, tw = target_dhw
    if (d, h, w) == (td, th, tw):
        return t
    out_shape = tuple(t.shape[:-3]) + target_dhw
    out = torch.zeros(out_shape, dtype=t.dtype, device=t.device)
    out[..., :d, :h, :w] = t
    return out


def scatter_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-size crop tensors to batch max spatial shape."""

    if not batch:
        return {}

    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        vals = [item[key] for item in batch]
        if key == "meta":
            out[key] = vals
            continue
        if torch.is_tensor(vals[0]):
            # Targets/present vectors have no spatial dims.
            if vals[0].ndim < 3:
                out[key] = torch.stack(vals, dim=0)
                continue
            max_d = max(int(v.shape[-3]) for v in vals)
            max_h = max(int(v.shape[-2]) for v in vals)
            max_w = max(int(v.shape[-1]) for v in vals)
            padded = [_pad_spatial_to(v, (max_d, max_h, max_w)) for v in vals]
            out[key] = torch.stack(padded, dim=0)
            continue
        out[key] = vals
    return out
