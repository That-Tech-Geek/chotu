"""Synthetic opponent used for benchmarking. Acts on the same economic ground
we claim to model — cost, reservation, patience, urgency — so the dining-room
rigour of the engine gets tested against an actual behaviour generator."""
from __future__ import annotations

import numpy as np


class SyntheticOpponent:
    """Supplier with latent economic state. Alternating-offers bargaining:
    accepts if the offer beats its reservation *markup floor given the
    pressure the round imposes; counters with a price withdrawing from the
    opening ask proportionally to patience/urgency; walks away on sustained
    lowball."""

    def __init__(
        self,
        supply_cost: float,
        reservation: float,
        patience: float = 0.5,
        urgency: float = 0.5,
        ask_markup: float = 0.25,
        rng: np.random.Generator | None = None,
    ):
        self.cost = supply_cost
        self.reservation = reservation
        self.patience = patience
        self.urgency = urgency
        self.ask = float(rng.uniform(1.05, 1.35)) * reservation if ask_markup is None else reservation * (1 + ask_markup)
        self.round = 0
        self.rng = rng or np.random.default_rng()

    def respond(self, offer_price: float) -> str:
        time_pressure = self.round * self.urgency / 8
        effective_reservation = self.reservation * (1 - 0.3 * min(time_pressure, 0.6))
        if offer_price >= effective_reservation:
            accept_p = min(
                0.95,
                0.5
                + (offer_price - effective_reservation) / max(self.reservation, 1e-3)
                + time_pressure,
            )
            if self.rng.random() < accept_p:
                return "ACCEPT"
        walk_p = min(0.02 + 0.04 * self.round * (1 - self.patience), 0.4)
        if offer_price < self.reservation * (1 - 0.1) and self.rng.random() < walk_p:
            return "WALKAWAY"
        return "COUNTER"

    def counter_offer(self, offer_price: float) -> float:
        progress = min(0.15 + (1 - self.patience) * 0.1 * self.round, 0.7)
        desired = (
            self.ask * (1 - progress)
            + self.reservation * progress
        )
        return float(max(self.reservation * 0.98, desired))
