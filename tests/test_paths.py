from __future__ import annotations

from scatterrad.paths import ScatterRadPaths


def test_paths_resolution(toy_dataset):
    paths = ScatterRadPaths.from_env(toy_dataset["dataset"])
    assert paths.dataset_json.exists()
    assert paths.targets_json.exists()
    assert paths.images_tr.exists()
    assert paths.labels_tr.exists()
    assert paths.crop_path("case001", 20).name == "case001_label020.npz"
