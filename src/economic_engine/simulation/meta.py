"""Meta-policy: pick the best strategy from {Chotu + all baselines} given
expected value. Also upgrade Chotu engine to include a surplus/aggressiveness
bias alongside risk aversion, rather than pure risk minimization.

This answers 'pick the best of them all' — the composite should hit held-out
with higher surplus than any single baseline while keeping deal-rate near
the best Chotu achieves."""
from __future__ import annotations

import numpy as np

from economic_engine.negotiation.engine import NegotiationEngine
from economic_engine.simulation.benchmark import (
    BaselinePolicyFactory,
    ChotuPolicy,
    _PolicyAdapter,
)
from economic_engine.simulation.opponent import SyntheticOpponent


class EnsemblePolicy:
    """Evaluates each candidate strategy's expected utility on the spot and
    returns the winning offer. Purchasing-adjusted eagerness balances
    surplus with the negative cost of a walkaway."""

    def __init__(
        self,
        risk_lambda: float = 0.5,
        walk_cost: float = 5.0,
        surplus_weight: float = 0.5,
    ):
        self.baselines = {
            "fixed": BaselinePolicyFactory.fixed_price,
            "concession": BaselinePolicyFactory.concession,
            "tft": BaselinePolicyFactory.tit_for_tat,
            "nash": BaselinePolicyFactory.nash,
            "chotu": ChotuPolicy,
        }
        self.risk_lambda = risk_lambda
        self.walk_cost = walk_cost
        self.surplus_weight = surplus_weight
        self.rng = np.random.default_rng(0)

    def next_offer(self, ctx, current_offer):
        candidates = {}
        for name, factory in self.baselines.items():
            if name == "chotu":
                candidates[name] = factory().next_offer(ctx, current_offer)
            else:
                candidates[name] = _PolicyAdapter(factory()).next_offer(
                    ctx, current_offer,
                )
        candidates = {k: v for k, v in candidates.items() if v is not None}
        if not candidates:
            return None
        # Expected score: balance surplus gain and risk / walk-cost trade.
        scores = {}
        for name, price in candidates.items():
            gain = self.surplus_weight * float(price)
            p_walk = self._estimate_walkaway(ctx, price)
            scores[name] = gain - self.walk_cost * p_walk
        winner = min(scores, key=scores.get)
        return candidates[winner]

    def _estimate_walkaway(self, ctx, price: float) -> float:
        """Rough walkaway hazard below the canonical supplier-reservation prior
        ~0.85·base, softened by how much below that we are."""
        supplier_reservation = ctx.product.base_purchase_cost * 0.85
        if price < supplier_reservation:
            excess = (supplier_reservation - price) / supplier_reservation
            return min(excess * 0.6, 0.5)
        return 0.05


def run_headtohead(n_holdout: int = 40, seed: int = 1000) -> dict:
    from economic_engine.simulation.benchmark import (
        BaselinePolicyFactory,
        ChotuPolicy,
        _PolicyAdapter,
        make_ctx,
        run_negotiation,
    )
    policies = {
        "ensemble": lambda: EnsemblePolicy(),
        "chotu": lambda: ChotuPolicy(),
        "nash": lambda: _PolicyAdapter(BaselinePolicyFactory.nash()),
        "fixed": lambda: _PolicyAdapter(BaselinePolicyFactory.fixed_price()),
    }
    results = {k: {"surplus": [], "deals": [], "rounds": []} for k in policies}
    for i in range(n_holdout):
        seed_i = seed + i
        base = float(np.random.default_rng(seed_i).uniform(80, 120))
        supplier_reservation = float(
            np.random.default_rng(seed_i + 1).uniform(0.75, 0.95) * base,
        )
        for name, factory in policies.items():
            ctx = make_ctx(base=base, seed=seed_i)
            opp = SyntheticOpponent(
                supply_cost=supplier_reservation * 0.75,
                reservation=supplier_reservation,
                patience=float(np.random.default_rng(seed_i + 2).uniform(0.3, 0.9)),
                urgency=float(np.random.default_rng(seed_i + 3).uniform(0.0, 1.0)),
                rng=np.random.default_rng(seed_i + 4),
            )
            policy = factory()
            r = run_negotiation(policy, ctx, opp, base)
            results[name]["surplus"].append(r["surplus"])
            results[name]["deals"].append(r["accepted"])
            if r["accepted"]:
                results[name]["rounds"].append(r["rounds"])
    return {
        name: {
            "deal_rate": float(np.mean(r["deals"])),
            "avg_surplus": float(np.mean(r["surplus"])),
            "avg_rounds": (
                float(np.mean(r["rounds"])) if r["rounds"] else float("nan")
            ),
        }
        for name, r in results.items()
    }
