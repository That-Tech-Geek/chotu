"""Safety gates for money actions: freshness -> margin -> limits -> risk ->
authorization -> audit. Result: ALLOWED / REQUIRES_APPROVAL / BLOCKED."""
from __future__ import annotations


class PolicyGate:
    def evaluate(
        self,
        *,
        transaction_value: float,
        discount_pct: float,
        risk_cvar: float,
        offer_price: float,
        constraints,
        data_age_seconds: float = 0.0,
    ) -> str:
        if data_age_seconds > constraints.data_freshness_seconds:
            return "BLOCKED"
        if constraints.max_offer is not None and offer_price > constraints.max_offer:
            return "BLOCKED"
        if (
            constraints.max_transaction_limit is not None
            and transaction_value > constraints.max_transaction_limit
        ):
            return "BLOCKED"
        if discount_pct > constraints.max_discount_pct:
            return "BLOCKED"
        if (
            constraints.requires_approval_above is not None
            and transaction_value > constraints.requires_approval_above
        ):
            return "REQUIRES_APPROVAL"
        return "ALLOWED"
