from __future__ import annotations

from scatterrad.config import PlansConfig


def test_plans_roundtrip(tmp_path):
    plans = PlansConfig(
        version=1,
        dataset_name="Toy",
        modality="CT",
        target_spacing_mm=(1.0, 1.0, 1.0),
        crop_size_voxels=(64, 64, 64),
        crop_margin_mm=10.0,
        intensity_clip=(-1000.0, 1000.0),
        intensity_mean=0.0,
        intensity_std=1.0,
        orientation="RAS",
        label_coverage={20: 3},
        bbox_percentiles={20: {"p50": (12, 12, 12), "p95": (16, 16, 16)}},
    )
    path = tmp_path / "plans.json"
    plans.to_json(path)
    loaded = PlansConfig.from_json(path)
    assert loaded == plans
