"""Relationship engine: LTV-aware deals for returning suppliers/merchants."""
from __future__ import annotations

from economic_engine.negotiation.opponent import OpponentState
from economic_engine.negotiation.strategies import Candidate
from economic_engine.state.canonical import NegotiationContext, Relationship


class RelationshipEngine:
    def estimate_relationship_value(self, rel: Relationship) -> float:
        # Expected future surplus: reputation + history increase LTV.
        return rel.lifetime_value + rel.reputation * 0.2 + rel.interaction_count * 0.05

    def personalize(
        self,
        ctx: NegotiationContext,
        candidate: Candidate,
        opponent,
    ) -> Candidate:
        rel = ctx.relationship
        if rel is None:
            return candidate
        val = self.estimate_relationship_value(rel)
        reservation = getattr(opponent.theta, "reservation_price", 1.0)
        if candidate.price is not None:
            candidate.price = max(
                candidate.price - 0.02 * val * abs(candidate.price),
                reservation,
            )
        return candidate
