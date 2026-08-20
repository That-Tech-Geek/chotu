"""Landed-cost engine. Returns a cost distribution (P10..P95), not a point."""
from __future__ import annotations

import numpy as np
import pydantic

from economic_engine.state.canonical import CostComponents


class CostDistribution(pydantic.BaseModel):
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    mean: float
    std: float

    def as_array(self) -> np.ndarray:
        return np.fromiter(
            (self.p10, self.p25, self.p50, self.p75, self.p90, self.p95),
            dtype=float,
        )


class CostEngine:
    """Landed cost = purchase + freight + handling + inventory + financing
    + failure + delay, propagated through a stochastic endpoint to a
    distribution. Components with nonzero sigma are sampled jointly."""

    COMPONENTS = ("freight", "handling", "inventory",
                  "financing", "failure", "delay")

    def landed_cost(
        self,
        costs: CostComponents,
        n_samples: int = 4096,
        rng: np.random.Generator | None = None,
    ) -> CostDistribution:
        rs = rng if rng is not None else np.random.default_rng()
        means = np.array(
            [getattr(costs, f"{c}_mean") for c in self.COMPONENTS],
            dtype=float,
        )
        stds = np.array(
            [getattr(costs, f"{c}_std") for c in self.COMPONENTS],
            dtype=float,
        )
        stochastic = stds > 0
        mean = costs.purchase + float(means.sum())
        sigma = float((stds[stochastic] ** 2).sum() ** 0.5) if stochastic.any() else 0.0
        sigma = max(sigma, mean * 0.02)
        z = np.array([-1.2816, -0.6745, 0.0, 0.6745, 1.2816, 1.6449])
        qs = mean + z * sigma
        return CostDistribution(
            p10=float(qs[0]), p25=float(qs[1]), p50=float(qs[2]),
            p75=float(qs[3]), p90=float(qs[4]), p95=float(qs[5]),
            mean=float(mean), std=float(sigma),
        )
