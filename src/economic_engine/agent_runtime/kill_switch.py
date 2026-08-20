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
        min_confidence: float = 0.3,
    ):
        self.max_price = max_price
        self.max_rounds = max_rounds
        self.min_confidence = min_confidence

    def evaluate(self, deal: Deal, ctx: NegotiationContext) -> str | None:
        """Return 'STOP: reason' or None."""
        if deal.price is not None and deal.price > self.max_price:
            return "STOP: price > max_price"
        if deal.quantity is not None and isinstance(deal.quantity, float):
            if ctx.negotiation.quantity and deal.quantity != ctx.negotiation.quantity:
                return "STOP: quantity changed"
        if ctx.negotiation.rounds and len(ctx.negotiation.rounds) > self.max_rounds:
            return "STOP: too many rounds"
        if ctx.relationship is not None and len(ctx.supplier.id) == 0:
            return "STOP: supplier identity uncertainty"
        return None

    def state_hash(self, ctx: NegotiationContext) -> str:
        raw = f"{ctx.merchant.id}:{ctx.product.id}:{ctx.negotiation.id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
