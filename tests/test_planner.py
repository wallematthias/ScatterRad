from __future__ import annotations

from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.planner import plan


def test_planner_writes_plans(toy_dataset):
    paths = ScatterRadPaths.from_env(toy_dataset["dataset"])
    plans = plan(paths)
    assert paths.plans_json.exists()
    assert plans.crop_size_voxels[0] % 8 == 0
    assert plans.label_coverage[20] == 5
