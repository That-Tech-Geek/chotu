"""Feature builder: canonical context -> numeric vector for the decision
engine and for storing feature snapshots."""
from __future__ import annotations

import numpy as np

from economic_engine.state.canonical import NegotiationContext

NAMES = [
    "listed_price", "landed_mean", "landed_spread", "round_index",
    "supplier_reliability", "relationship_value", "acceptance_hint",
    "inventory_available", "demand_mean", "demand_std",
    "sentiment", "urgency", "price_resistance", "concession_willingness",
    "deadline_signal", "uncertainty", "finality", "relationship_signal",
    "lead_time", "lead_time_std",
]


def build_features(ctx: NegotiationContext) -> dict[str, float]:
    neg = ctx.negotiation
    last_offer = neg.rounds[-1].offer if neg.rounds else None
    price = last_offer.price if last_offer and last_offer.price else ctx.product.base_purchase_cost
    cost_mean = 0.0
    cost_spread = 0.0
    if ctx.costs is not None:
        from economic_engine.models.cost_engine import CostEngine
        dist = CostEngine().landed_cost(ctx.costs, n_samples=512)
        cost_mean = dist.mean
        cost_spread = dist.p95 - dist.p10
    sig = ctx.text_signals
    inv = ctx.inventory.available if ctx.inventory else 0.0
    dem_mean = ctx.demand.mean if ctx.demand else 0.0
    dem_std = ctx.demand.std if ctx.demand else 0.0
    rel_val = ctx.relationship.lifetime_value if ctx.relationship else 0.0
    lead = ctx.logistics.lead_time_days if ctx.logistics else 0.0
    lead_std = ctx.logistics.lead_time_std if ctx.logistics else 0.0
    return {
        "listed_price": price or 0.0,
        "landed_mean": cost_mean,
        "landed_spread": cost_spread,
        "round_index": float(len(neg.rounds)),
        "supplier_reliability": ctx.supplier.reliability_history,
        "relationship_value": rel_val,
        "acceptance_hint": 0.5,
        "inventory_available": inv,
        "demand_mean": dem_mean,
        "demand_std": dem_std,
        "sentiment": sig.sentiment if sig else 0.0,
        "urgency": sig.urgency if sig else 0.0,
        "price_resistance": sig.price_resistance if sig else 0.0,
        "concession_willingness": sig.concession_willingness if sig else 0.0,
        "deadline_signal": sig.deadline_signal if sig else 0.0,
        "uncertainty": sig.uncertainty if sig else 0.0,
        "finality": sig.finality if sig else 0.0,
        "relationship_signal": sig.relationship_signal if sig else 0.0,
        "lead_time": lead,
        "lead_time_std": lead_std,
    }


def to_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features.get(k, 0.0) for k in NAMES], dtype=np.float32)
