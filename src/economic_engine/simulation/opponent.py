"""Synthetic opponent used for benchmarking. Acts on the same economic ground
we claim to model — cost, reservation, patience, urgency — so the dining-room
rigour of the engine gets tested against an actual behaviour generator."""
from __future__ import annotations

import numpy as np


class SyntheticOpponent:
    def __init__(
        self,
        supply_cost: float,
        reservation: float,
        patience: float = 0.5,
        urgency: float = 0.5,
        discount: float = 0.95,
        rng: np.random.Generator | None = None,
    ):
        self.cost = supply_cost
        self.reservation = reservation
        self.patience = patience
        self.urgency = urgency
        self.discount = discount
        self.round = 0
        self.rng = rng or np.random.default_rng()

    def respond(self, offer_price: float) -> str:
        """Return ACCEPT/COUNTER/WALKAWAY for the price."""
        margin = offer_price - self.reservation
        time_pressure = self.round * (1 - self.patience) / 20
        urgency = self.urgency * 0.1 + time_pressure
        if margin < -self.reservation * 0.05 + urgency * self.reservation * 0.1:
            walk_prob = min(0.05 + (1 - self.patience) * 0.1 * self.round, 0.5)
            if self.rng.random() < walk_prob:
                self.round += 1
                return "WALKAWAY"
        if margin >= 0.0:
            p_accept = min(0.9, 0.4 + margin / max(self.reservation, 1e-3)
                           + urgency)
            if self.rng.random() < p_accept:
                self.round += 1
                return "ACCEPT"
        else:
            ask = self.reservation * (1 + 0.05 * (1 - self.patience))
            if offer_price >= ask:
                self.round += 1
                return "ACCEPT"
        self.round += 1
        return "COUNTER"

    def counter_offer(self, offer_price: float, discount_this_round: float) -> float:
        move = 0.5 * (1 - self.patience) * discount_this_round
        return (offer_price + self.reservation) / 2 * (1 - move * 0.1)
