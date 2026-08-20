"""Explicit game-theoretic solver: alternating-offers bargaining with
utility functions U_B(p,t,s) and U_S(p,t) driven by opponent posterior theta.

This replaces the old 'estimate acceptance -> map to walk/counter' heuristic;
we now evaluate candidate actions against an explicit strategic model of the
opponent and maximize combined sequential utility."""
from __future__ import annotations

import math

import numpy as np

from economic_engine.negotiation.opponent import OpponentState
from economic_engine.negotiation.strategies import Candidate
from economic_engine.state.canonical import NegotiationContext


class BargainingSolver:
    def __init__(self, horizon: int = 5):
        self.horizon = horizon

    @staticmethod
    def buyer_utility(
        price: float,
        landed_cost: float,
        quantity: float,
        discount: float,
        risk: float = 0.0,
    ) -> float:
        surplus = landed_cost - price
        return quantity * surplus * discount - risk

    @staticmethod
    def supplier_utility(
        price: float,
        cost: float,
        quantity: float,
        discount: float,
    ) -> float:
        surplus = price - cost
        return quantity * surplus * discount

    def response_prob_matrix(
        self,
        ctx: NegotiationContext,
        candidates: list[Candidate],
        opponent: OpponentState,
    ) -> np.ndarray:
        """P(response | action) as a matrix over candidates; not just scalar."""
        probs = np.zeros((len(candidates), 3))
        for i, cand in enumerate(candidates):
            p = cand.price
            if p is None:
                p = (
                    ctx.negotiation.rounds[-1].offer.price
                    if ctx.negotiation.rounds
                    else ctx.product.base_purchase_cost
                )
            if p is None:
                p = 1.0
            accept = opponent.response_probability(float(p))
            walk = opponent.theta.walkaway_probability * (1 - accept)
            counter = max(1 - accept - walk, 0.0)
            probs[i] = [accept, counter, walk]
        return probs

    def solve(
        self,
        ctx: NegotiationContext,
        candidates: list[Candidate],
        opponent: OpponentState,
    ) -> tuple[int, float]:
        """Pick the action maximizing sequential expected utility over the
        horizon, conditioning on opponent theta."""
        probs = self.response_prob_matrix(ctx, candidates, opponent)
        best_idx, best_utility = 0, -float("inf")
        quantity = ctx.negotiation.quantity
        delta = opponent.theta.discount_factor
        for i, cand in enumerate(candidates):
            p = cand.price
            if p is None:
                p = (
                    ctx.negotiation.rounds[-1].offer.price
                    if ctx.negotiation.rounds
                    else ctx.product.base_purchase_cost
                )
            if p is None:
                p = 1.0
            p = float(p)
            ev = 0.0
            for r in range(1, self.horizon + 1):
                half_r = (r - 1) / 2 if r > 1 else 0
                u_b = self.buyer_utility(
                    p,
                    ctx.product.base_purchase_cost,
                    quantity,
                    discount=delta ** r,
                )
                u_s = self.supplier_utility(
                    p,
                    opponent.theta.cost_structure,
                    quantity,
                    discount=delta ** r,
                )
                ev += probs[i][0] * (u_b + u_s) * delta ** half_r
            if float(ev) > best_utility:
                best_utility = float(ev)
                best_idx = i
        return best_idx, best_utility
