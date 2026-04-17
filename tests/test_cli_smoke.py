from __future__ import annotations

from scatterrad.cli import main


def test_cli_validate_smoke(toy_dataset):
    code = main(["validate", toy_dataset["dataset"]])
    assert code == 0


def test_cli_plan_preprocess_smoke(toy_dataset):
    ds = toy_dataset["dataset"]
    assert main(["plan", "--dataset", ds, "--planner", "RadiomicsPlanner"]) == 0
    assert main(["preprocess", "--dataset", ds, "--num-workers", "0"]) == 0


def test_cli_generate_holdout_smoke(toy_dataset):
    ds = toy_dataset["dataset"]
    assert main(["generate-holdout", "--dataset", ds, "--fraction", "0.2"]) == 0
