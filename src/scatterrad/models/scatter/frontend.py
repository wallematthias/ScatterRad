from __future__ import annotations

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Wavelet sub-bands (SWT)
# ---------------------------------------------------------------------------

_SUBBAND_KEYS_3D = ["aaa", "aad", "ada", "add", "daa", "dad", "dda", "ddd"]


def _swt3_numpy(volume: np.ndarray, wavelet: str = "coif1", level: int = 1) -> np.ndarray:
    """3-D stationary wavelet transform → (C, D, H, W) float32."""
    import pywt

    orig_shape = volume.shape
    pad = tuple((0, 1 if s % 2 != 0 else 0) for s in orig_shape)
    needs_pad = any(p[1] for p in pad)
    if needs_pad:
        volume = np.pad(volume, pad, mode="wrap")

    dec_levels = pywt.swtn(volume, pywt.Wavelet(wavelet), level=level, axes=(0, 1, 2))
    channels = []
    for lvl_idx, dec in enumerate(dec_levels):
        is_last = lvl_idx == len(dec_levels) - 1
        for key in _SUBBAND_KEYS_3D:
            if key == "aaa" and not is_last:
                continue
            band = dec[key].astype(np.float32, copy=False)
            if needs_pad:
                band = band[tuple(slice(None, s) for s in orig_shape)]
            channels.append(band)
    return np.stack(channels, axis=0)  # (C, D, H, W)


def _n_wavelet_channels(level: int) -> int:
    return 7 * level + 1


# ---------------------------------------------------------------------------
# LoG and gradient filters (SimpleITK, already a project dependency)
# ---------------------------------------------------------------------------

def _log_numpy(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    sigmas_mm: tuple[float, ...],
) -> np.ndarray:
    """Laplacian-of-Gaussian at multiple scales → (len(sigmas), D, H, W) float32."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(volume.astype(np.float32))
    img.SetSpacing(tuple(reversed(spacing_mm)))  # SimpleITK: (x, y, z)

    channels = []
    filt = sitk.LaplacianRecursiveGaussianImageFilter()
    filt.SetNormalizeAcrossScale(True)
    for sigma in sigmas_mm:
        filt.SetSigma(float(sigma))
        out = sitk.GetArrayFromImage(filt.Execute(img)).astype(np.float32)
        channels.append(out)
    return np.stack(channels, axis=0)  # (S, D, H, W)


def _gradient_numpy(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    """Gradient magnitude → (1, D, H, W) float32."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(volume.astype(np.float32))
    img.SetSpacing(tuple(reversed(spacing_mm)))
    filt = sitk.GradientMagnitudeImageFilter()
    filt.SetUseImageSpacing(True)
    out = sitk.GetArrayFromImage(filt.Execute(img)).astype(np.float32)
    return out[None]  # (1, D, H, W)


# ---------------------------------------------------------------------------
# Second-order (GLCM-derived) statistics per channel
# ---------------------------------------------------------------------------

_N_GLCM_STATS = 4  # energy, contrast, homogeneity, correlation


def _glcm_features_1d(values: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """GLCM statistics from a 1-D array of voxel values inside the ROI.

    Returns [energy, contrast, homogeneity, correlation].
    """
    if values.size < 4:
        return np.zeros(4, dtype=np.float32)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax == vmin:
        return np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    bins = np.linspace(vmin, vmax, n_bins + 1)
    idx = np.digitize(values, bins[1:-1]).astype(np.int32)
    i_vals, j_vals = idx[:-1], idx[1:]
    glcm = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(glcm, (i_vals, j_vals), 1)
    np.add.at(glcm, (j_vals, i_vals), 1)
    total = glcm.sum()
    if total == 0:
        return np.zeros(4, dtype=np.float32)
    glcm /= total

    ii = np.arange(n_bins, dtype=np.float64)
    I, J = np.meshgrid(ii, ii, indexing="ij")

    energy = float((glcm ** 2).sum())
    contrast = float(((I - J) ** 2 * glcm).sum())
    homogeneity = float((glcm / (1.0 + np.abs(I - J))).sum())
    mu_i = float((I * glcm).sum())
    mu_j = float((J * glcm).sum())
    std_i = float(np.sqrt(((I - mu_i) ** 2 * glcm).sum()))
    std_j = float(np.sqrt(((J - mu_j) ** 2 * glcm).sum()))
    correlation = (
        0.0 if (std_i < 1e-8 or std_j < 1e-8)
        else float(((I - mu_i) * (J - mu_j) * glcm / (std_i * std_j)).sum())
    )
    return np.array([energy, contrast, homogeneity, correlation], dtype=np.float32)


def _second_order_features(bands: np.ndarray, mask: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """GLCM stats for each channel, restricted to mask.

    bands: (C, D, H, W); mask: (D, H, W) binary
    Returns (C * 4,) float32.
    """
    C = bands.shape[0]
    out = np.zeros(C * _N_GLCM_STATS, dtype=np.float32)
    fg = mask > 0.5
    if fg.sum() < 4:
        return out
    for c in range(C):
        vals = bands[c][fg].astype(np.float32)
        out[c * _N_GLCM_STATS:(c + 1) * _N_GLCM_STATS] = _glcm_features_1d(vals, n_bins=n_bins)
    return out


# ---------------------------------------------------------------------------
# Combined filter bank
# ---------------------------------------------------------------------------

def _build_filter_bank(
    volume: np.ndarray,           # (D, H, W) float32, already masked if mask_mode='zero'
    spacing_mm: tuple[float, float, float],
    wavelet: str,
    wavelet_level: int,
    log_sigmas_mm: tuple[float, ...],
    use_gradient: bool,
) -> np.ndarray:
    """Compute all filter-bank channels for one volume.

    Returns (C_total, D, H, W) float32 where:
        C_total = wavelet_channels + len(log_sigmas_mm) + (1 if use_gradient else 0)
    """
    parts = [_swt3_numpy(volume, wavelet=wavelet, level=wavelet_level)]
    if log_sigmas_mm:
        parts.append(_log_numpy(volume, spacing_mm, log_sigmas_mm))
    if use_gradient:
        parts.append(_gradient_numpy(volume, spacing_mm))
    return np.concatenate(parts, axis=0)  # (C, D, H, W)


def _n_filter_channels(wavelet_level: int, log_sigmas_mm: tuple, use_gradient: bool) -> int:
    # +1 for the original image channel prepended in forward()
    return 1 + _n_wavelet_channels(wavelet_level) + len(log_sigmas_mm) + (1 if use_gradient else 0)


# ---------------------------------------------------------------------------
# WaveletFrontend (now a full filter-bank frontend)
# ---------------------------------------------------------------------------

class WaveletFrontend(nn.Module):
    """Fixed 3-D filter-bank frontend.

    Computes per-voxel feature maps from three complementary filter families:

    * **Wavelet sub-bands** (SWT, coif1 level 1 → 8 channels):
      Low/high-pass decomposition along each axis pair — captures orientation-
      specific frequency content, the same basis PyRadiomics uses.

    * **Laplacian of Gaussian** (optional, one channel per σ):
      Blob/edge detector at multiple physical scales.  σ=1 mm → fine texture
      (trabecular bone detail); σ=3 mm → coarser lesion-scale changes.

    * **Gradient magnitude** (optional, 1 channel):
      Local edge strength |∇I|, sensitive to boundary sharpness.

    The output is a (B, C, D, H, W) volume at the same spatial resolution as
    the input — no downsampling.  The backend then extracts scalar statistics
    per channel inside the ROI mask (first-order mean + second-order GLCM).
    """

    def __init__(
        self,
        crop_size: tuple[int, int, int],
        spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        wavelet: str = "coif1",
        level: int = 1,
        log_sigmas_mm: tuple[float, ...] = (1.0, 2.0, 3.0),
        use_gradient: bool = True,
        mask_mode: str = "zero",
        out_channels_override: int | None = None,
        out_shape_override: tuple | None = None,
    ):
        super().__init__()
        if mask_mode not in {"zero", "post_pool"}:
            raise ValueError("mask_mode must be 'zero' or 'post_pool'")
        self.crop_size = crop_size
        self.spacing_mm = spacing_mm
        self.wavelet = wavelet
        self.level = level
        self.log_sigmas_mm = tuple(log_sigmas_mm)
        self.use_gradient = use_gradient
        self.mask_mode = mask_mode
        # Legacy aliases
        self.J = level
        self.L = 0

        if out_channels_override is not None and out_shape_override is not None:
            self._out_channels = int(out_channels_override)
            self._out_shape = tuple(out_shape_override)
        else:
            self._out_channels = _n_filter_channels(level, self.log_sigmas_mm, use_gradient)
            self._out_shape = tuple(crop_size)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def out_shape(self) -> tuple[int, int, int]:
        return self._out_shape

    def forward(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (filter_maps, mask).

        filter_maps: (B, 1 + C_filters, D, H, W)
            Channel 0:    original (normalised) image — passed through so the
                          conv backend can learn raw-intensity vs filter-response
                          relationships.
            Channels 1…:  wavelet sub-bands, LoG images, gradient magnitude.
        mask: (B, 1, D, H, W) — unchanged.
        """
        device = image.device
        # Zero-mask the image for filter computation
        x = image * mask if self.mask_mode == "zero" else image

        x_np = x.detach().cpu().numpy()  # (B, 1, D, H, W)

        results = []
        for i in range(x_np.shape[0]):
            vol = x_np[i, 0]
            filters = _build_filter_bank(
                vol,
                spacing_mm=self.spacing_mm,
                wavelet=self.wavelet,
                wavelet_level=self.level,
                log_sigmas_mm=self.log_sigmas_mm,
                use_gradient=self.use_gradient,
            )
            # Prepend original image as channel 0
            combined = np.concatenate([vol[None], filters], axis=0)  # (1+C, D, H, W)
            results.append(combined)

        out = np.stack(results, axis=0)  # (B, 1+C, D, H, W)
        return torch.from_numpy(out).to(device=device, dtype=torch.float32), mask


# Alias for backward compatibility
ScatterFrontend = WaveletFrontend
