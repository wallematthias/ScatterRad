from __future__ import annotations

import json
from pathlib import Path

from scatterrad.config import ModelKind, load_task_config, load_targets_schema
from scatterrad.paths import ScatterRadPaths


def test_task_config(toy_dataset, tmp_path: Path):
    paths = ScatterRadPaths.from_env(toy_dataset["dataset"])
    schema = load_targets_schema(paths.targets_json)
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps(
            {
                "name": "fracture_scatter_v1",
                "target": "fracture",
                "model": "scatter",
                "cv": {"folds": 3, "seed": 7},
            }
        )
    )
    task = load_task_config(task_path, schema=schema)
    assert task.model is ModelKind.SCATTER
    assert task.resolved_labels(schema) == (20, 21)
