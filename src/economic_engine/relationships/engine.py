"""Relationship engine: LTV-aware deals for returning suppliers/merchants."""
from __future__ import annotations

from economic_engine.models.supplier_model import SupplierPosterior
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
        posterior: SupplierPosterior,
    ) -> Candidate:
        rel = ctx.relationship
        if rel is None:
            return candidate
        val = self.estimate_relationship_value(rel)
        if candidate.price is not None:
            candidate.price = max(
                candidate.price - 0.02 * val * abs(candidate.price),
                posterior.reservation_mean,
            )
        return candidate
