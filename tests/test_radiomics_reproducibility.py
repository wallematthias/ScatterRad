from __future__ import annotations

import numpy as np

from scatterrad.models.radiomics.reproducibility import _icc_1_1


def test_icc_1_1_high_for_consistent_repeats():
    x = np.array(
        [
            [1.0, 1.02, 0.98],
            [2.0, 2.01, 1.99],
            [3.0, 3.03, 2.97],
            [4.0, 4.01, 3.99],
        ]
    )
    icc = _icc_1_1(x)
    assert icc > 0.95


def test_icc_1_1_nan_for_invalid_shape():
    x = np.array([1.0, 2.0, 3.0])
    icc = _icc_1_1(x)
    assert np.isnan(icc)
