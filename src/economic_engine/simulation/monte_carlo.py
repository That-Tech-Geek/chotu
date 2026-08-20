"""Vectorized Monte Carlo over candidate actions, sampling supplier, price,
quantity, demand, cost, logistics, failure and relationship outcomes."""
from __future__ import annotations

import numpy as np

from economic_engine.models.cost_engine import CostEngine
from economic_engine.negotiation.strategies import Candidate
from economic_engine.state.canonical import NegotiationContext

FAST_STANDARD_DEEP = {"FAST": 4096, "STANDARD": 32768, "DEEP": 131072}


class MonteCarloSimulator:
    def __init__(self, mode: str = "STANDARD", seed: int = 0):
        self.mode = mode
        self.n = FAST_STANDARD_DEEP.get(mode, 32768)
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        ctx: NegotiationContext,
        candidate: Candidate,
    ) -> np.ndarray:
        """Return a vector of profit samples; vectorized across samples."""
        n = self.n
        neg = ctx.negotiation
        offered = candidate.price
        if offered is None and neg.rounds:
            offered = neg.rounds[-1].offer.price
        if offered is None:
            offered = ctx.product.base_purchase_cost
        if ctx.costs is not None:
            cd = CostEngine().landed_cost(ctx.costs, n_samples=n, rng=self.rng)
            cost_mean, cost_std = cd.mean, cd.std
        else:
            cost_mean, cost_std = offered, 0.0
        supplier_reliability = ctx.supplier.reliability_history
        delay_mean = ctx.logistics.lead_time_days if ctx.logistics else 0.0
        delay_penalty = delay_mean * 0.005
        cost = self.rng.normal(cost_mean, cost_std, n)
        dem_mean = ctx.demand.mean if ctx.demand else 1.0
        dem_std = ctx.demand.std if ctx.demand else 0.5
        demand_factor = np.clip(self.rng.normal(dem_mean, dem_std + 1e-6, n), 0, None)
        profit_per_unit = offered - cost
        penalty = delay_penalty * (1 - supplier_reliability)
        penalty_abs = abs(profit_per_unit.min()) if profit_per_unit.size else 1.0
        rel_value = (
            ctx.relationship.lifetime_value if ctx.relationship else 0.0
        )
        relation_bonus = rel_value * 0.01
        profit = profit_per_unit * demand_factor + relation_bonus - penalty * penalty_abs * 0.01
        return profit
