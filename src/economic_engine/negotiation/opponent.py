"""First-class OpponentState: latent behavioural parameters (not just an
acceptance curve). The posterior over theta is the core economic moat; each
round updates P(theta | D_{1:t}) and downstream utilities condition on it."""
from __future__ import annotations

import math

import numpy as np


class OpponentLatent:
    def __init__(
        self,
        reservation_price: float,
        reservation_std: float,
        cost_structure: float | None = None,
        urgency: float = 0.5,
        margin_requirement: float = 0.15,
        patience: float = 0.5,
        discount_factor: float = 0.95,
        walkaway_probability: float = 0.2,
        counteroffer_elasticity: float = 0.5,
        quantity_sensitivity: float = 0.5,
        relationship_value: float = 0.5,
        information_sensitivity: float = 0.5,
        anchoring_sensitivity: float = 0.5,
        deadline_sensitivity: float = 0.5,
    ):
        self.reservation_price = reservation_price
        self.reservation_std = reservation_std
        self.cost_structure = cost_structure or reservation_price * 0.7
        self.urgency = urgency
        self.margin_requirement = margin_requirement
        self.patience = patience
        self.discount_factor = discount_factor
        self.walkaway_probability = walkaway_probability
        self.counteroffer_elasticity = counteroffer_elasticity
        self.quantity_sensitivity = quantity_sensitivity
        self.relationship_value = relationship_value
        self.information_sensitivity = information_sensitivity
        self.anchoring_sensitivity = anchoring_sensitivity
        self.deadline_sensitivity = deadline_sensitivity
        # Confidence grows as observations accumulate (0..1).
        self.confidence = 0.2


class OpponentState:
    """Bayesian state of the supplier. Not a scalar acceptance curve."""

    def __init__(self, latent: OpponentLatent, history: list[dict] | None = None):
        self.theta = latent
        self.history = history or []

    def confidence(self) -> float:
        return min(self.theta.confidence + 0.02 * len(self.history), 1.0)

    def response_probability(self, price: float) -> float:
        gap = price - self.theta.reservation_price
        z = gap / max(self.theta.reservation_std, 1e-3)
        base = 0.5 + 0.35 * math.tanh(z)
        accept = min(max(base, 0.05), 0.95)
        # Anchoring behaviour makes concessions more likely above reservation.
        if price > self.theta.reservation_price:
            accept += 0.1 * self.theta.anchoring_sensitivity
        # High urgency & low patience increases acceptance of the tabled price.
        accept += 0.05 * self.theta.urgency - 0.03 * self.theta.patience
        return float(min(max(accept, 0.05), 0.97))

    def expected_payoff(self, price: float, round_num: int) -> float:
        margin = max(price - self.theta.reservation_price, 0.0)
        cost = self.theta.cost_structure
        accept = self.response_probability(price)
        discount = self.theta.discount_factor ** round_num
        return accept * margin * discount - (1 - accept) * cost * 0.02

    def update_from_round(
        self,
        price: float,
        accepted: bool,
        signals=None,
    ) -> None:
        """Posterior update: theta -> theta' given the observed response."""
        self.history.append({"price": price, "accepted": accepted})
        if accepted:
            self.theta.reservation_price = (
                0.6 * self.theta.reservation_price
                + 0.4 * price
            )
            self.theta.reservation_std = max(self.theta.reservation_std * 0.92, 0.01)
            self.theta.walkaway_probability = max(
                self.theta.walkaway_probability - 0.05, 0.02,
            )
        else:
            direction = -1 if price < self.theta.reservation_price else 1
            self.theta.reservation_price += direction * 0.4 * self.theta.reservation_std
            self.theta.reservation_std = min(
                self.theta.reservation_std * 1.03,
                self.theta.reservation_price * 0.5,
            )
            self.theta.urgency = min(self.theta.urgency + 0.05, 1.0)
        if signals is not None:
            self.theta.anchoring_sensitivity = min(
                max(self.theta.anchoring_sensitivity + 0.1 * signals.finality, 0.0), 1.0,
            )
            self.theta.deadline_sensitivity = min(
                max(self.theta.deadline_sensitivity + 0.1 * signals.deadline_signal, 0.0), 1.0,
            )
            self.theta.patience = min(max(self.theta.patience - 0.05 * signals.urgency, 0.0), 1.0)
