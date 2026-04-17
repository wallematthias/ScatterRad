from __future__ import annotations

from collections import Counter

import numpy as np
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler[int]):
    """Class-balanced oversampling sampler.

    Reads targets directly from dataset.samples to avoid loading crops at init.
    Re-seeds per epoch so each epoch gets a different shuffled order.
    """

    def __init__(
        self,
        dataset,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        self.seed = seed
        self.num_samples = num_samples or len(dataset)
        self._epoch = 0

        target_type = dataset.target_type.value
        if target_type == "regression":
            self.weights = np.ones(len(dataset), dtype=np.float64)
            self.weights /= self.weights.sum()
            return

        # Read targets from case_targets without loading crops
        from scatterrad.config import TargetScope
        values = []
        for sample in dataset.samples:
            if dataset.scope is TargetScope.PER_LABEL:
                basename, label_id = sample
                v = dataset.case_targets[basename].get_per_label(dataset.task.target, label_id)
            else:
                basename = sample
                v = dataset.case_targets[basename].get_per_case(dataset.task.target)
            values.append(int(v))

        counts = Counter(values)
        self.weights = np.asarray([1.0 / counts[v] for v in values], dtype=np.float64)
        self.weights /= self.weights.sum()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        indices = rng.choice(len(self.weights), size=self.num_samples, replace=True, p=self.weights)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples
