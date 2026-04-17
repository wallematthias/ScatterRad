from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk


def _write_nifti(path: Path, arr: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> None:
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing(tuple(reversed(spacing)))
    sitk.WriteImage(image, str(path))


@pytest.fixture()
def toy_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path | str]:
    dataset = "DatasetToy"

    raw_root = tmp_path / "raw"
    pre_root = tmp_path / "pre"
    res_root = tmp_path / "res"

    raw_dir = raw_root / dataset
    images = raw_dir / "imagesTr"
    labels = raw_dir / "labelsTr"
    targets = raw_dir / "targetsTr"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    targets.mkdir(parents=True)

    dataset_json = {
        "name": dataset,
        "modality": {"0": "CT"},
        "labels": {"0": "background", "20": "L1", "21": "L2"},
        "numTraining": 3,
        "file_ending": ".nii.gz",
    }
    (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")

    targets_json = {
        "fracture": {
            "type": "classification",
            "scope": "per_label",
            "num_classes": 2,
            "applicable_labels": [20, 21],
        },
        "source_level": {
            "type": "classification",
            "scope": "per_case",
            "num_classes": 5,
            "applicable_labels": [20, 21],
        },
        "age": {
            "type": "regression",
            "scope": "per_case",
            "applicable_labels": [20, 21],
        },
    }
    (raw_dir / "targets.json").write_text(json.dumps(targets_json, indent=2) + "\n")

    for i in range(5):
        name = f"case{i + 1:03d}"
        image = np.zeros((24, 24, 24), dtype=np.float32)
        label = np.zeros((24, 24, 24), dtype=np.uint8)
        image[6:14, 6:14, 6:14] = 300 + i
        label[7:12, 7:12, 7:12] = 20
        if i != 1:
            image[12:20, 12:20, 12:20] = -400
            label[13:18, 13:18, 13:18] = 21
        _write_nifti(images / f"{name}_0000.nii.gz", image)
        _write_nifti(labels / f"{name}.nii.gz", label)

        t = {
            "age": 40 + i,
            "source_level": i + 1,
            "fracture": {"20": int(i % 2), "21": int((i + 1) % 2)},
        }
        (targets / f"{name}.json").write_text(json.dumps(t, indent=2) + "\n")

    monkeypatch.setenv("SCATTERRAD_RAW", str(raw_root))
    monkeypatch.setenv("SCATTERRAD_PREPROCESSED", str(pre_root))
    monkeypatch.setenv("SCATTERRAD_RESULTS", str(res_root))
    monkeypatch.setenv("SCATTERRAD_RAD_PERTURB_N", "1")
    monkeypatch.setenv("SCATTERRAD_RAD_PERTURB_MAX_CASES", "8")
    monkeypatch.setenv("SCATTERRAD_RAD_PERTURB_SEED", "42")

    return {
        "dataset": dataset,
        "raw": raw_root,
        "pre": pre_root,
        "res": res_root,
    }
