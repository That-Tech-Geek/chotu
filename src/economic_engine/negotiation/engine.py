"""Core negotiate loop: observe -> model -> generate -> simulate -> optimize.
Uses OpponentState (latent behavioural posterior), BargainingSolver, Monte
Carlo, CVaR, risk-adjusted aggressiveness, sim-estimated VOI, and gates."""
from __future__ import annotations

import numpy as np

from economic_engine.models.cost_engine import CostEngine
from economic_engine.negotiation.opponent import OpponentLatent, OpponentState
from economic_engine.negotiation.solver import BargainingSolver
from economic_engine.negotiation.strategies import generate_candidates
from economic_engine.optimization.information import ValueOfInformation
from economic_engine.optimization.objectives import cvar, fractional_kelly
from economic_engine.policy.gates import PolicyGate
from economic_engine.relationships.engine import RelationshipEngine
from economic_engine.simulation.monte_carlo import MonteCarloSimulator
from economic_engine.state.canonical import NegotiationContext


class NegotiationEngine:
    def __init__(
        self,
        opponent: OpponentState | None = None,
        mc_mode: str = "STANDARD",
        exposure_capital: float = 100_000.0,
        type_prior=None,
        lambda_override: float | None = None,
    ):
        self.lambda_override = lambda_override
        self.type_prior = type_prior
        self.opponent = opponent or OpponentState(
            OpponentLatent(reservation_price=1.0, reservation_std=0.15)
        )
        self.mc = MonteCarloSimulator(mc_mode)
        self.solver = BargainingSolver()
        self.cost = CostEngine()
        self.relationship = RelationshipEngine()
        self.gate = PolicyGate()
        self.voi = ValueOfInformation()
        self.exposure_capital = exposure_capital

    def decide(self, ctx: NegotiationContext) -> dict:
        neg = ctx.negotiation
        if self.opponent.theta.reservation_price == 1.0:
            base = ctx.product.base_purchase_cost or 1.0
            if self.type_prior is not None:
                self.opponent = self.type_prior.initial_opponent(
                    ctx_base=base,
                    supplier_type=(
                        ctx.product.category if ctx.product else "generic"
                    ),
                )
            else:
                self.opponent = OpponentState(
                    OpponentLatent(
                        reservation_price=base,
                        reservation_std=base * 0.15,
                        cost_structure=base * 0.7,
                        relationship_value=(
                            ctx.relationship.reputation if ctx.relationship else 0.5
                        ),
                    )
                )
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
            current_price=last_price or landed_mean or reservation,
            landed_mean=landed_mean or 1.0,
            reservation_price=reservation,
            quantity=neg.quantity,
        )
        personalized = [
            self.relationship.personalize(ctx, cand, self.opponent)
            for cand in candidates
        ]
        scores = []
        for cand in personalized:
            profits = self.mc.simulate(ctx, cand)
            cvar_95 = cvar(profits, alpha=0.95)
            expected = float(np.mean(profits))
            price_eval = cand.price or last_price or landed_mean or 1.0
            p_accept = self.opponent.response_probability(float(price_eval))
            # Kelly here is capital-allocation against negotiation exposure:
            # it bounds aggressiveness relative to exposure capital.
            kelly = fractional_kelly(
                win_prob=max(p_accept, 0.01),
                win_return=max(abs(expected), 1.0) / self.exposure_capital,
                fraction=0.25,
            )
            lambda_ = (
                self.lambda_override
                if self.lambda_override is not None
                else max(1.0 - kelly, 0.2)
            )
            utility = expected - lambda_ * abs(cvar_95)
            scores.append({
                "candidate": cand,
                "utility": float(utility),
                "expected": float(expected),
                "cvar_95": float(cvar_95),
                "p_accept": float(p_accept),
            })
        utils = np.array([s["utility"] for s in scores])
        solver_idx, _ = self.solver.solve(ctx, personalized, self.opponent)
        best = (
            scores[solver_idx]
            if utils[solver_idx] >= utils.max() * 0.95
            else scores[int(np.argmax(utils))]
        )
        # VOI estimated from candidate utilities: would asking a question
        # change the decision?
        util_list = [s["utility"] for s in scores]
        voi = self.voi.estimate(
            current_utility=float(best["utility"]),
            candidate_questions={"info": sorted(util_list, reverse=True)[:3]},
            cost_of_asking=0.05 * abs(best["utility"]),
        )
        if voi > 0:
            ask = next(
                (c for c in personalized if c.action.value == "ASK_INFORMATION"),
                None,
            )
            if ask is not None:
                best = next(s for s in scores if s["candidate"] is ask)
        price = (
            best["candidate"].price
            or (neg.rounds[-1].offer.price if neg.rounds else ctx.product.base_purchase_cost)
        )
        return {
            "action": best["candidate"].action.value,
            "strategy": best["candidate"].strategy.value,
            "price": round(float(price or 0), 2),
            "expected_profit": int(round(float(best["expected"]), 0)),
            "acceptance_probability": round(float(best["p_accept"]), 3),
            "cvar_95": round(float(best["cvar_95"]), 3),
            "confidence": round(
                float(min(0.99, 0.5 + 0.5 * best["p_accept"]
                          * self.opponent.confidence())), 2),
            "voi": round(float(voi), 3),
            "reason_codes": [
                f"UTILITY={best['utility']:.2f}",
                f"LANDED_MEAN={landed_mean:.3f}",
                f"CVAR95={best['cvar_95']:.2f}",
                f"OPP_CONF={self.opponent.confidence():.2f}",
            ],
        }

    def update_from_round(
        self,
        ctx: NegotiationContext,
        proposed_price: float,
        response: str,
        signals=None,
    ) -> None:
        self.opponent.update_from_round(
            price=proposed_price,
            accepted=response.upper() == "ACCEPT",
            signals=signals,
        )
