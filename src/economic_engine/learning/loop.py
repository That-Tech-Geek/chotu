"""Learning loop: decision -> outcome -> error -> metrics -> fitness ->
calibration -> candidate -> replay/shadow -> promote. Never in-request."""
from __future__ import annotations

import numpy as np


class LearningLoop:
    def prediction_error(self, predicted: float, actual: float) -> float:
        return float(actual - predicted)

    def calibration(self, errors: list[float]) -> float:
        if not errors:
            return 0.0
        arr = np.asarray(errors, dtype=float)
        return float(np.mean(arr ** 2) ** 0.5)

    def should_promote(
        self,
        candidate_error: float,
        current_error: float,
        shadow_n: int,
        min_shadow_n: int = 200,
    ) -> bool:
        if shadow_n < min_shadow_n or current_error <= 0:
            return False
        return candidate_error < 0.95 * current_error

    def record_outcome(self, record: dict, store) -> None:
        store.append(record)
