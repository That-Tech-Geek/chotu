"""PolicyEnvelope — the hard box the optimizer plays inside. The brain can be
creative in here; it cannot edit the box itself."""
from __future__ import annotations

from economic_engine.state.canonical import Deal


class PolicyEnvelope:
    def __init__(
        self,
        max_unit_price: float,
        min_unit_price: float,
        max_total_spend: float,
        min_margin: float = 0.0,
        max_quantity: float | None = None,
        allowed_payment_terms: list[str] | None = None,
        allowed_delivery_window: tuple[int, int] | None = None,
        max_rounds: int = 30,
        max_negotiation_seconds: int = 86400,
    ):
        self.max_unit_price = max_unit_price
        self.min_unit_price = min_unit_price
        self.max_total_spend = max_total_spend
        self.min_margin = min_margin
        self.max_quantity = max_quantity
        self.allowed_payment_terms = allowed_payment_terms
        self.allowed_delivery_window = allowed_delivery_window
        self.max_rounds = max_rounds
        self.max_negotiation_seconds = max_negotiation_seconds

    def check_deal(self, deal: Deal, base_cost: float | None = None) -> list[str]:
        violations: list[str] = []
        if deal.price is not None:
            if deal.price > self.max_unit_price:
                violations.append("price exceeds max_unit_price")
            if deal.price < self.min_unit_price:
                violations.append("price below min_unit_price")
        if base_cost is not None and deal.price is not None:
            margin = (base_cost - deal.price) / base_cost if base_cost > 0 else 0.0
            if margin < self.min_margin:
                violations.append("margin below min_margin")
        if deal.quantity is not None:
            if self.max_quantity is not None and deal.quantity > self.max_quantity:
                violations.append("quantity exceeds envelope")
            if deal.price is not None:
                total = deal.price * deal.quantity
                if total > self.max_total_spend:
                    violations.append("total spend exceeds cap")
        if deal.payment_terms and self.allowed_payment_terms is not None:
            if deal.payment_terms not in self.allowed_payment_terms:
                violations.append("payment_terms not allowed")
        if deal.delivery_window_days and self.allowed_delivery_window is not None:
            lo, hi = self.allowed_delivery_window
            if not (lo <= deal.delivery_window_days <= hi):
                violations.append("delivery window outside allowed range")
        return violations
