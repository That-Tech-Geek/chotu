"""Estimated value of information for candidate next questions."""
from __future__ import annotations

import numpy as np


class ValueOfInformation:
    """VOI(a) = E_x[max_d EU(d|x)] - max_d EU(d) - cost(a)."""

    def __init__(self, n_samples: int = 512):
        self.n = n_samples

    def estimate(
        self,
        current_utility: float,
        candidate_questions: dict[str, list[float]],
        cost_of_asking: float,
    ) -> float:
        if not candidate_questions:
            return -float(cost_of_asking)
        max_future = max(
            self._best_utility_posterite(v) for v in candidate_questions.values()
        )
        return float(max_future - current_utility - cost_of_asking)

    def _best_utility_posterite(self, utils: list[float]) -> float:
        arr = np.asarray(utils, dtype=float)
        return float(arr.max())
