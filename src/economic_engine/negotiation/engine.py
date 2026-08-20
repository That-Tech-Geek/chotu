"""Core negotiate loop: observe -> model -> generate -> simulate -> optimize.
Uses posterior supplier model, game planner, Monte Carlo, CVaR/Kelly and
policy gates. All numpy, no LLM."""
from __future__ import annotations

import numpy as np

from economic_engine.models.cost_engine import CostEngine
from economic_engine.models.supplier_model import SupplierPosterior
from economic_engine.negotiation.game import GameTheoreticPlanner
from economic_engine.negotiation.strategies import (
    Action,
    Candidate,
    generate_candidates,
)
from economic_engine.optimization.objectives import (
    cvar,
    fractional_kelly,
    value_of_information,
)
from economic_engine.policy.gates import PolicyGate
from economic_engine.relationships.engine import RelationshipEngine
from economic_engine.simulation.monte_carlo import MonteCarloSimulator
from economic_engine.state.canonical import NegotiationContext


class NegotiationEngine:
    def __init__(
        self,
        posterior: SupplierPosterior | None = None,
        mc_mode: str = "STANDARD",
        studies: int = 6,
    ):
        self.posterior = posterior or SupplierPosterior()
        self.mc = MonteCarloSimulator(mc_mode)
        self.planner = GameTheoreticPlanner()
        self.cost = CostEngine()
        self.relationship = RelationshipEngine()
        self.gate = PolicyGate()

    def decide(self, ctx: NegotiationContext) -> dict:
        neg = ctx.negotiation
        last_price = (
            neg.rounds[-1].offer.price
            if neg.rounds
            else ctx.product.base_purchase_cost
        )
        landed = (
            self.cost.landed_cost(ctx.costs, n_samples=1024, rng=self.mc.rng)
            if ctx.costs is not None
            else None
        )
        landed_mean = landed.mean if landed else (ctx.product.base_purchase_cost or 1.0)
        reservation = neg.reservation_price
        candidates = generate_candidates(
            current_price=last_price or landed_mean,
            landed_mean=landed_mean,
            reservation_price=reservation,
            quantity=neg.quantity,
        )
        best = None
        best_utility = -np.inf
        scores = []
        for cand in candidates:
            per = self.relationship.personalize(ctx, cand, self.posterior)
            probs = self.planner.response_probabilities(ctx, per, self.posterior)
            profits = self.mc.simulate(ctx, per)
            cvar_95 = cvar(profits, alpha=0.95)
            expected = float(np.mean(profits))
            p_accept = probs.get("ACCEPT", 0.5)
            kelly = fractional_kelly(
                win_prob=max(p_accept, 0.01),
                win_return=max(abs(expected), 1.0),
                fraction=0.25,
            )
            lambda_ = max(1.0 - kelly, 0.2)
            utility = expected - lambda_ * abs(cvar_95)
            c_ui = ctx.text_signals
            voi = value_of_information(
                current_ev=utility,
                expected_ev_with_info=utility + 0.1,
                cost_of_asking=0.05,
            )
            scores.append(
                {
                    "candidate": per,
                    "utility": float(utility),
                    "expected": float(expected),
                    "cvar_95": float(cvar_95),
                    "p_accept": float(p_accept),
                    "voi": float(voi),
                }
            )
            if utility > best_utility:
                best_utility = utility
                best = scores[-1]
        assert best is not None
        price = (
            best["candidate"].price
            or (
                neg.rounds[-1].offer.price
                if neg.rounds
                else ctx.product.base_purchase_cost
            )
        )
        return {
            "action": best["candidate"].action.value,
            "strategy": best["candidate"].strategy.value,
            "price": round(float(price or 0), 2),
            "expected_profit": int(round(float(best["expected"]), 0)),
            "acceptance_probability": round(float(best["p_accept"]), 3),
            "cvar_95": round(float(best["cvar_95"]), 3),
            "confidence": round(float(min(0.99, 0.5 + 0.5 * float(best["p_accept"]))), 2),
            "reason_codes": [
                f"UTILITY={best['utility']:.2f}",
                f"LANDED_MEAN={landed_mean:.3f}",
                f"CVAR95={best['cvar_95']:.2f}",
            ],
        }

    def update_from_round(
        self,
        ctx: NegotiationContext,
        proposed_price: float,
        response: str,
        signals=None,
    ) -> None:
        self.posterior.update_from_round(
            round_index=len(ctx.negotiation.rounds) - 1,
            proposed_price=proposed_price,
            response=response,
            signals=signals,
        )
