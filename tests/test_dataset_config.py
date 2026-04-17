from __future__ import annotations

import json

from scatterrad.config import load_dataset_config
from scatterrad.paths import ScatterRadPaths


def test_load_dataset_config(toy_dataset):
    paths = ScatterRadPaths.from_env(toy_dataset["dataset"])
    cfg = load_dataset_config(paths.dataset_json)
    assert cfg.modality == "CT"
    assert cfg.label_ids == [20, 21]
    assert cfg.label_name(20) == "L1"


def test_load_dataset_config_accepts_nnunet_style(tmp_path):
    dataset_json = {
        "name": "DatasetNNUNetLike",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "T1": 1, "T2": 2},
    }
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(dataset_json) + "\n")
    cfg = load_dataset_config(p)
    assert cfg.modality == "CT"
    assert cfg.label_ids == [1, 2]
    assert cfg.label_name(1) == "T1"
