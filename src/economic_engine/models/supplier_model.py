"""Posterior supplier model P(theta|history). Analytic Gaussian/Beta updates
- no training loop in the request path."""
from __future__ import annotations

import math

import numpy as np

from economic_engine.state.canonical import Offer, TextSignals


class SupplierPosterior:
    def __init__(
        self,
        reservation_mean: float = 1.0,
        reservation_std: float = 0.15,
        acceptance_base: float = 0.5,
        concession_tendency: float = 0.5,
        patience: float = 0.5,
        deadline_sensitivity: float = 0.5,
        reliability: float = 0.5,
        batna: float = 0.5,          # how strong supplier alternatives are
        relationship_sensitivity: float = 0.5,
        price_sensitivity: float = 0.5,
    ):
        self.reservation_mean = reservation_mean
        self.reservation_std = max(reservation_std, 1e-3)
        self.acceptance_base = acceptance_base
        self.concession_tendency = concession_tendency
        self.patience = patience
        self.deadline_sensitivity = deadline_sensitivity
        self.reliability = reliability
        self.batna = batna
        self.relationship_sensitivity = relationship_sensitivity
        self.price_sensitivity = price_sensitivity

    def update_from_round(
        self,
        round_index: int,
        proposed_price: float,
        response: str,
        signals: TextSignals | None = None,
    ) -> None:
        """Bayesian-style posterior adjustment from one round outcome."""
        accepted = response.upper() == "ACCEPT"
        if accepted:
            delta = proposed_price - self.reservation_mean
            new = self.reservation_mean + 0.4 * delta
            self.reservation_std *= 0.85
            self.reservation_mean = new
            self.acceptance_base = min(self.acceptance_base + 0.1, 1.0)
        else:
            direction = 1.0 if proposed_price < self.reservation_mean else -1.0
            self.reservation_mean += direction * 0.3 * self.reservation_std
            self.reservation_std = min(self.reservation_std * 1.02, 1.0)
            self.acceptance_base = max(self.acceptance_base - 0.05, 0.05)
        if signals is not None:
            conf = abs(signals.sentiment)
            self.relationship_sensitivity = _clip(
                self.relationship_sensitivity + 0.2 * conf * (0.5 if signals.sentiment > 0 else -0.5)
            )
            self.concession_tendency = _clip(
                self.concession_tendency + 0.2 * signals.concession_willingness
            )
            self.deadline_sensitivity = _clip(
                self.deadline_sensitivity + 0.15 * signals.deadline_signal
            )
            self.patience = _clip(self.patience - 0.1 * signals.urgency)

    def acceptance_probability(self, price: float) -> float:
        gap = price - self.reservation_mean
        z = gap / max(self.reservation_std, 1e-3)
        base = 0.5 + 0.1 * math.tanh(z)
        accept = base - self.price_sensitivity * 0.2 + self.acceptance_base * 0.4
        return float(max(0.05, min(0.95, accept)))

    def expected_concession(self, current_price: float) -> float:
        return current_price - max(
            self.reservation_mean - self.reservation_std,
            self.reservation_mean * 0.6,
        ) * self.concession_tendency


def _clip(v: float) -> float:
    return float(max(0.0, min(1.0, v)))
