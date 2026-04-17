from __future__ import annotations

import json

from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.holdout import generate_holdout


def test_generate_holdout_moves_files_and_writes_manifest(toy_dataset):
    paths = ScatterRadPaths.from_env(toy_dataset["dataset"])
    payload = generate_holdout(paths, holdout_fraction=0.2, seed=42, move_files=True)

    assert len(payload["holdout_cases"]) == 1
    holdout_id = payload["holdout_cases"][0]
    assert (paths.raw_dataset_dir / "test_split_manifest.json").exists()
    assert not (paths.targets_tr / f"{holdout_id}.json").exists()
    assert (paths.targets_ts / f"{holdout_id}.json").exists()

    moved_images = payload["moved_to_ts"]["images"]["moved"]
    assert moved_images >= 1


def test_generate_holdout_no_move_does_not_modify_training_dirs(toy_dataset):
    paths = ScatterRadPaths.from_env(toy_dataset["dataset"])
    before = sorted(p.name for p in paths.targets_tr.glob("*.json"))

    payload = generate_holdout(paths, holdout_fraction=0.2, seed=42, move_files=False)
    after = sorted(p.name for p in paths.targets_tr.glob("*.json"))

    assert before == after
    manifest = json.loads((paths.raw_dataset_dir / "test_split_manifest.json").read_text())
    assert manifest["holdout_cases"] == payload["holdout_cases"]
