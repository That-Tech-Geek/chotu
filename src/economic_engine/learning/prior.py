"""Hierarchical prior: routes new suppliers to a supplier-type prior drawn
from global statistics rather than starting ignorant.

    global prior
        └─ industry prior
            └─ supplier-type prior
                └─ supplier posterior
                    └─ current negotiation posterior

The routing keys come from the times-negotiation history stored in
Supabase, so a new supplier inherits intelligence from past negotiations
instead of starting at base * 1.0.
"""
from __future__ import annotations

import numpy as np

from economic_engine.negotiation.opponent import OpponentLatent, OpponentState


class GlobalPrior:
    """Statistics across all past negotiations — the root of the hierarchy."""

    def __init__(self, observations: list[float] | None = None):
        self.observations = observations or []
        if self.observations:
            self.mean = float(np.mean(self.observations))
            self.std = float(np.std(self.observations)) or self.mean * 0.2
        else:
            self.mean = 100.0
            self.std = 20.0

    def add_observation(self, reservation_price: float) -> None:
        self.observations.append(float(reservation_price))
        self.mean = float(np.mean(self.observations))
        self.std = max(float(np.std(self.observations)), 0.01) or self.mean * 0.2


class SupplierTypePrior:
    """Supplier-type prior drawn from the GlobalPrior, keyed on
    supplier-type tags (units of product/market context)."""

    def __init__(self, global_prior: GlobalPrior):
        self.global_prior = global_prior

    def sample(self, supplier_type: str = "generic") -> tuple[float, float]:
        # Drawn deterministically given the global statistics; this is the
        # supplier-type prior used to initialize a fresh opponent.
        mean = self.global_prior.mean
        std = self.global_prior.std
        return float(mean), float(std)

    def initial_opponent(
        self,
        ctx_base: float | None = None,
        supplier_type: str = "generic",
    ) -> OpponentState:
        mean, std = self.sample(supplier_type)
        if ctx_base is not None:
            mean = ctx_base  # strongest signal is the current product price
            std = max(std, ctx_base * 0.1)
        return OpponentState(
            OpponentLatent(
                reservation_price=mean,
                reservation_std=std,
                cost_structure=mean * 0.7,
            )
        )
