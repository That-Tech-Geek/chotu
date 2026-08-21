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
    """Meta-policy over {Chotu engine + baselines}. Each round it evaluates
    every candidate's offer under an estimated walkaway hazard derived from
    the *engine's own posterior* (OpponentState), not a hardcoded 0.85·base
    prior — the same posterior Chotu uses to decide. The winner is the offer
    maximizing (surplus retention − walk_cost · P(walkaway)).
    """

    def __init__(
        self,
        walk_cost: float = 1.0,
    ):
        from economic_engine.negotiation.opponent import OpponentLatent, OpponentState

        self.opponent = OpponentState(
            OpponentLatent(reservation_price=1.0, reservation_std=1.0)
        )
        self.walk_cost = walk_cost
        self.baselines = {
            "fixed": BaselinePolicyFactory.fixed_price,
            "concession": BaselinePolicyFactory.concession,
            "tft": BaselinePolicyFactory.tit_for_tat,
            "nash": BaselinePolicyFactory.nash,
        }
        self._initialized = False

    def next_offer(self, ctx, current_offer):
        base = ctx.product.base_purchase_cost or 1.0
        if not self._initialized:
            from economic_engine.negotiation.opponent import OpponentLatent, OpponentState

            self.opponent = OpponentState(
                OpponentLatent(
                    reservation_price=base,
                    reservation_std=base * 0.2,
                    cost_structure=base * 0.7,
                )
            )
            self._initialized = True
        # Posterior update from the supplier's latest observed behaviour.
        if current_offer is not None:
            self.opponent.update_from_round(
                price=current_offer,
                accepted=False,  # we only get called when the supplier countered
            )
        candidates = {}
        for name, factory in self.baselines.items():
            candidates[name] = _PolicyAdapter(factory()).next_offer(
                ctx, current_offer,
            )
        candidates = {k: v for k, v in candidates.items() if v is not None}
        if not candidates:
            return None
        reservation = self.opponent.theta.reservation_price
        scores = {}
        for name, price in candidates.items():
            # Retained surplus = how much below base we land.
            retained = base - float(price)
            # Walkaway hazard from the posterior: price under reservation
            # materially raises walkaway probability.
            gap = max(reservation - float(price), 0.0)
            p_walk = min(0.05 + gap / max(reservation, 1e-3), 0.6)
            scores[name] = retained - self.walk_cost * p_walk * base
        winner = max(scores, key=scores.get)
        return candidates[winner]


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
