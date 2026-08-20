"""Benchmark: engine vs baselines over synthetic negotiations. Answers the
question — 'does Chotu beat simple strategies at the thing it claims to do?'"""
from __future__ import annotations

import numpy as np

from economic_engine.negotiation.engine import NegotiationEngine
from economic_engine.simulation.opponent import SyntheticOpponent
from economic_engine.state.canonical import (
    CostComponents,
    Merchant,
    Negotiation,
    NegotiationContext,
    NegotiationStatus,
    Offer,
    Product,
    Relationship,
    Round,
    Supplier,
)


def make_ctx(
    merchant_id: str,
    base: float,
    seed: int,
    qty: float = 10,
    reservation_price: float | None = None,
) -> NegotiationContext:
    rng = np.random.default_rng(seed)
    base_cost = float(base)
    return NegotiationContext(
        merchant=Merchant(id=merchant_id, name="m"),
        supplier=Supplier(id=f"s{seed}", merchant_id=merchant_id, name="s"),
        product=Product(id=f"p{seed}", merchant_id=merchant_id,
                        sku=f"sku{seed}", base_purchase_cost=base_cost),
        negotiation=Negotiation(
            id=f"n{seed}", merchant_id=merchant_id, supplier_id=f"s{seed}",
            product_id=f"p{seed}", quantity=qty, status=NegotiationStatus.OPEN,
        ),
        costs=CostComponents(
            purchase=base_cost,
            freight_mean=float(rng.uniform(0.02, 0.08) * base_cost),
            freight_std=float(rng.uniform(0.0, 0.03) * base_cost),
            handling_mean=float(rng.uniform(0.005, 0.01) * base_cost),
        ),
        relationship=Relationship(
            merchant_id=merchant_id, supplier_id=f"s{seed}",
            interaction_count=0, reputation=0.5, lifetime_value=0.0,
        ),
    )


class FixedPricePolicy:
    def next_offer(self, ctx, current_offer: float) -> float | None:
        return current_offer


class LinearConcessionPolicy:
    def __init__(self, step: float = 0.05):
        self.step = step

    def next_offer(self, ctx, current_offer: float) -> float | None:
        return current_offer * (1 + self.step)


class TitForTatPolicy:
    def next_offer(self, ctx, current_offer: float) -> float | None:
        return current_offer * 0.93


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def next_offer(self, ctx, current_offer: float) -> float | None:
        return current_offer * self.rng.uniform(0.8, 1.2)


class NashHeuristicPolicy:
    def next_offer(self, ctx, current_offer: float) -> float | None:
        supplier_reservation = ctx.product.base_purchase_cost * 0.85
        return (current_offer + supplier_reservation) / 2


class ChotuPolicy:
    def __init__(self):
        self.engine = NegotiationEngine(mc_mode="FAST")

    def next_offer(
        self,
        ctx: NegotiationContext,
        current_offer: float,
    ) -> float | None:
        ctx.negotiation.rounds.append(
            Round(index=0, offer=Offer(price=current_offer, actor="supplier"))
        )
        decision = self.engine.decide(ctx)
        if decision["action"] == "WALKAWAY":
            return None
        return float(decision["price"])


def run_negotiation(
    policy,
    ctx: NegotiationContext,
    opponent: SyntheticOpponent,
    base: float,
    max_rounds: int = 7,
) -> dict:
    current_offer = base * 1.2
    for rnd in range(1, max_rounds + 1):
        response = opponent.respond(current_offer)
        if response == "ACCEPT":
            price = current_offer
            return {
                "accept": True,
                "price": price,
                "rounds": rnd,
                "surplus": base - price,
            }
        if response == "WALKAWAY":
            return {"accept": False, "price": None, "rounds": rnd, "surplus": 0.0}
        next_offer = policy.next_offer(ctx, current_offer=current_offer)
        if next_offer is None:
            return {"accept": False, "price": None, "rounds": rnd, "surplus": 0.0}
        current_offer = next_offer
    price = current_offer
    if opponent.respond(price) == "ACCEPT":
        return {
            "accept": True,
            "price": price,
            "rounds": max_rounds,
            "surplus": base - price,
        }
    return {"accept": False, "price": None, "rounds": max_rounds, "surplus": 0.0}


def benchmark(
    n: int = 300,
    policies: dict | None = None,
    seed: int = 0,
) -> dict:
    if policies is None:
        policies = {"chotu": ChotuPolicy()}
    results: dict[str, list] = {k: [] for k in policies}
    for i in range(n):
        seed_i = seed + i
        base = float(np.random.default_rng(seed_i).uniform(80, 120))
        supplier_reservation = base * float(np.random.default_rng(seed_i + 1).uniform(0.75, 0.95))
        supply_cost = supplier_reservation * 0.75
        for name, policy in policies.items():
            ctx = make_ctx(merchant_id="merchant", base=base, seed=seed_i)
            opp = SyntheticOpponent(
                supply_cost=supply_cost,
                reservation=supplier_reservation,
                patience=float(np.random.default_rng(seed_i + 2).uniform(0.3, 0.9)),
                urgency=float(np.random.default_rng(seed_i + 3).uniform(0.0, 1.0)),
                rng=np.random.default_rng(seed_i + 4),
            )
            results[name].append(run_negotiation(policy, ctx, opp, base))
    agg = {}
    for name, rs in results.items():
        arr = np.asarray([r["surplus"] for r in rs])
        wins = [r["accept"] for r in rs]
        agg[name] = {
            "avg_surplus": float(arr.mean()),
            "wins": float(np.mean(wins)),
            "avg_rounds": float(np.mean([r["rounds"] for r in rs])),
        }
    return agg
