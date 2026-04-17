from __future__ import annotations

import json

from scatterrad.paths import ScatterRadPaths
from scatterrad.preprocessing.planner import plan
from scatterrad.preprocessing.runner import preprocess


def test_preprocess_writes_crops_and_splits(toy_dataset):
    paths = ScatterRadPaths.from_env(toy_dataset["dataset"])
    _ = plan(paths)
    preprocess(paths)

    crops = sorted(paths.crops_dir.glob("*.npz"))
    assert crops
    payload = json.loads(paths.splits_json.read_text())
    assert payload["n_folds"] == 5
    assert len(payload["folds"]) == 5
    assert paths.results_splits_json.exists()
    assert (paths.preprocessed_dataset_dir / "radiomics_reproducibility" / "icc_scores.json").exists()
    assert (paths.preprocessed_dataset_dir / "radiomics_analysis" / "summary.json").exists()
    assert paths.preprocessed_dataset_json.exists()
    assert paths.preprocessed_targets_json.exists()
    first_target = json.loads((paths.preprocessed_targets_tr / "case001.json").read_text())
    assert first_target["source_level"] == 0
