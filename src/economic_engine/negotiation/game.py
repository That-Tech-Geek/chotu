"""Approximate alternating-offers model: P(response|state,action) via the
supplier posterior, reservation values, BATNA, discount factors."""
from __future__ import annotations

import math

from economic_engine.models.supplier_model import SupplierPosterior
from economic_engine.negotiation.strategies import Candidate
from economic_engine.state.canonical import NegotiationContext


class GameTheoreticPlanner:
    def __init__(
        self,
        discount: float = 0.95,   # per-round discount factor
        information_asymmetry: float = 1.0,
    ):
        self.discount = discount
        self.information_asymmetry = information_asymmetry

    def response_probabilities(
        self,
        ctx: NegotiationContext,
        candidate: Candidate,
        posterior: SupplierPosterior,
    ) -> dict[str, float]:
        price = candidate.price
        if price is None and ctx.negotiation.rounds:
            price = ctx.negotiation.rounds[-1].offer.price
        if price is None:
            price = ctx.product.base_purchase_cost
        accept = posterior.acceptance_probability(price)
        adjust = 1.0
        if candidate.action in ("WAIT", "ASK_INFORMATION"):
            adjust *= 0.7
        if candidate.action == "WALKAWAY":
            accept = 0.0
        counter = max(1.0 - accept, 0.0) * 0.8 * adjust
        walk = max(1.0 - accept - counter, 0.0)
        return {"ACCEPT": accept * adjust, "COUNTER": counter, "WALKAWAY": walk}

    def nash_quotient(self, our_offer: float, their_reservation: float) -> float:
        """Simple Nash-bargaining split heuristic: mid between batna-anchored
        bounds. Higher is closer to their reservation price."""
        spread = our_offer - their_reservation
        return math.exp(-abs(spread) / max(our_offer, 1.0))
