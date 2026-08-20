"""Deterministic kill switches. STOP means stop; the engine-in-flight
executor never retries an overridden command."""
from __future__ import annotations

import hashlib

from economic_engine.state.canonical import Deal, NegotiationContext


class KillSwitch:
    def __init__(
        self,
        max_price: float,
        max_rounds: int = 30,
        min_identity_confidence: float = 0.3,
    ):
        self.max_price = max_price
        self.max_rounds = max_rounds
        self.min_identity_confidence = min_identity_confidence

    def evaluate(self, deal: Deal, ctx: NegotiationContext) -> str | None:
        """Return 'STOP: reason' or None."""
        if deal.price is not None and deal.price > self.max_price:
            return "STOP: price > max_price"
        if deal.quantity is not None:
            if ctx.negotiation.quantity and deal.quantity != ctx.negotiation.quantity:
                return "STOP: quantity changed"
        if ctx.negotiation.rounds and len(ctx.negotiation.rounds) > self.max_rounds:
            return "STOP: too many rounds"
        # Identity confidence: empty supplier id is a hard stop; else if the
        # relationship carries a confidence that is below the threshold, we stop.
        if not ctx.supplier.id:
            return "STOP: supplier identity uncertain"
        if ctx.relationship is not None:
            id_conf = getattr(ctx.relationship, "identity_confidence", None)
            if id_conf is not None and id_conf < self.min_identity_confidence:
                return "STOP: identity confidence below threshold"
        return None

    def state_hash(self, ctx: NegotiationContext) -> str:
        raw = f"{ctx.merchant.id}:{ctx.product.id}:{ctx.negotiation.id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
