from __future__ import annotations

import SimpleITK as sitk


def _as_ras(image: sitk.Image) -> sitk.Image:
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("RAS")
    return orient.Execute(image)


def resample_to_spacing(
    image: sitk.Image,
    target_spacing: tuple[float, float, float],
    is_label: bool,
) -> sitk.Image:
    """Resample to target spacing in RAS orientation."""

    src = _as_ras(image)
    old_spacing = src.GetSpacing()
    old_size = src.GetSize()
    new_size = [
        max(1, int(round(sz * sp / tsp)))
        for sz, sp, tsp in zip(old_size, old_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(src.GetOrigin())
    resampler.SetOutputDirection(src.GetDirection())
    resampler.SetTransform(sitk.Transform())
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        default_value = 0
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
        default_value = 0.0
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(src)
