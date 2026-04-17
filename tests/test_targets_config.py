from __future__ import annotations

from scatterrad.config import load_case_targets, load_targets_schema
from scatterrad.paths import ScatterRadPaths


def test_targets_schema_and_case_targets(toy_dataset):
    paths = ScatterRadPaths.from_env(toy_dataset["dataset"])
    schema = load_targets_schema(paths.targets_json, known_label_ids={20, 21})
    case = load_case_targets(paths.targets_tr / "case001.json", schema)
    assert schema["fracture"].num_classes == 2
    assert case.get_per_label("fracture", 20) in {0.0, 1.0}
    assert case.get_per_case("age") == 40.0
