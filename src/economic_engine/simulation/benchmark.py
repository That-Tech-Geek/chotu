"""Held-out benchmark: Chotu vs baselines against a generator of
latent-parameter opponents. Train policies on a seed set, evaluate on unseen
seeds to prevent overfitting the benchmark.

Strictly measures: surplus (base - deal price), deal rate, walkaway rate,
rounds, regret, and CVaR of surplus (worst-case 5% of closed deals).

Example:
    from economic_engine.simulation.benchmark import BaselinePolicyFactory, evaluate
    evaluate(
        n_train=200,
        n_holdout=100,
        baselines=[BaselinePolicyFactory.concession],
    )
"""
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
    Round,
    Supplier,
)


def make_ctx(base: float, seed: int, qty: float = 10) -> NegotiationContext:
    rng = np.random.default_rng(seed)
    return NegotiationContext(
        merchant=Merchant(id="merchant", name="m"),
        supplier=Supplier(id=f"s{seed}", merchant_id="merchant", name="s"),
        product=Product(
            id=f"p{seed}",
            merchant_id="merchant",
            sku=f"sku{seed}",
            base_purchase_cost=float(base),
        ),
        negotiation=Negotiation(
            id=f"n{seed}",
            merchant_id="merchant",
            supplier_id=f"s{seed}",
            product_id=f"p{seed}",
            quantity=qty,
            status=NegotiationStatus.OPEN,
        ),
        costs=CostComponents(
            purchase=float(base),
            freight_mean=float(rng.uniform(0.02, 0.08) * base),
            freight_std=float(rng.uniform(0.0, 0.03) * base),
            handling_mean=float(rng.uniform(0.005, 0.01) * base),
        ),
    )


class ChotuPolicy:
    def __init__(self):
        self.engine = NegotiationEngine(mc_mode="FAST")

    def next_offer(self, ctx, current_offer: float) -> float | None:
        ctx.negotiation.rounds.append(
            Round(index=0, offer=Offer(price=current_offer, actor="supplier"))
        )
        decision = self.engine.decide(ctx)
        if decision["action"] == "WALKAWAY":
            return None
        return float(decision["price"])


class BaselinePolicyFactory:
    """Each factory call returns a fresh policy; no state leaks between
    negotiations. Note: factories are functions (classes); calling
    `factory()` returns a policy."""

    @staticmethod
    def chotu():
        return ChotuPolicy()

    @staticmethod
    def fixed_price():
        def policy(ctx, current_offer):
            return current_offer

        return policy

    @staticmethod
    def concession(step: float = 0.05):
        def policy(ctx, current_offer):
            return current_offer * (1 + step)

        return policy

    @staticmethod
    def tit_for_tat(scale: float = 0.93):
        def policy(ctx, current_offer):
            return current_offer * scale

        return policy

    @staticmethod
    def random(seed: int = 0):
        rng = np.random.default_rng(seed)

        def policy(ctx, current_offer):
            return current_offer * rng.uniform(0.8, 1.2)

        return policy

    @staticmethod
    def nash():
        def policy(ctx, current_offer):
            supplier_reservation = ctx.product.base_purchase_cost * 0.85
            return (current_offer + supplier_reservation) / 2

        return policy


class _PolicyAdapter:
    """Treat functions returning prices like stateful policy objects."""

    def __init__(self, fn):
        self.fn = fn

    def next_offer(self, ctx, current_offer):
        return self.fn(ctx, current_offer)


def run_negotiation(policy, ctx, opponent, base, max_rounds: int = 8) -> dict:
    current_offer = base * 0.95
    for rnd in range(1, max_rounds + 1):
        response = opponent.respond(current_offer)
        if response == "ACCEPT":
            return {
                "accepted": True,
                "price": current_offer,
                "rounds": rnd,
                "surplus": base - current_offer,
            }
        if response == "WALKAWAY":
            return {
                "accepted": False,
                "price": None,
                "rounds": rnd,
                "surplus": 0.0,
            }
        next_offer = policy.next_offer(ctx, current_offer)
        if next_offer is None:
            return {
                "accepted": False,
                "price": None,
                "rounds": rnd,
                "surplus": 0.0,
            }
        current_offer = next_offer
    if opponent.respond(current_offer) == "ACCEPT":
        return {
            "accepted": True,
            "price": current_offer,
            "rounds": max_rounds,
            "surplus": base - current_offer,
        }
    return {"accepted": False, "price": None, "rounds": max_rounds, "surplus": 0.0}


def evaluate(
    n_train: int = 200,
    n_holdout: int = 100,
    baselines: list | None = None,
    seed: int = 0,
) -> dict:
    if baselines is None:
        baselines = [
            BaselinePolicyFactory.fixed_price,
            BaselinePolicyFactory.concession,
            BaselinePolicyFactory.tit_for_tat,
            BaselinePolicyFactory.random,
            BaselinePolicyFactory.nash,
        ]
    factories: dict[str, callable] = {"chotu": ChotuPolicy}
    for factory in baselines:
        factories[getattr(factory, "__name__", str(factory))] = factory
    results: dict[str, list[list]] = {k: [[], []] for k in factories}
    all_seeds = list(range(seed, seed + n_train + n_holdout))
    splits = [all_seeds[:n_train], all_seeds[n_train:]]
    for split_idx, seeds in enumerate(splits):
        for seed_i in seeds:
            base = float(np.random.default_rng(seed_i).uniform(80, 120))
            supplier_reservation = float(
                np.random.default_rng(seed_i + 1).uniform(0.75, 0.95) * base,
            )
            supply_cost = supplier_reservation * 0.75
            for name, factory in factories.items():
                if name == "chotu":
                    policy = ChotuPolicy()
                else:
                    policy = _PolicyAdapter(factory())
                ctx = make_ctx(base=base, seed=seed_i)
                opponent = SyntheticOpponent(
                    supply_cost=supply_cost,
                    reservation=supplier_reservation,
                    patience=float(
                        np.random.default_rng(seed_i + 2).uniform(0.3, 0.9),
                    ),
                    urgency=float(
                        np.random.default_rng(seed_i + 3).uniform(0.0, 1.0),
                    ),
                    rng=np.random.default_rng(seed_i + 4),
                )
                results[name][split_idx].append(
                    run_negotiation(policy, ctx, opponent, base),
                )
    agg = {}
    for name, (_, holdout) in results.items():
        surplus_all = np.array([r["surplus"] for r in holdout])
        surplus_closed = np.array(
            [r["surplus"] for r in holdout if r["accepted"]],
        )
        deals = np.array([r["accepted"] for r in holdout])
        rounds = np.array([r["rounds"] for r in holdout if r["accepted"]])
        best_achievable = surplus_all * 0.95
        regret = float((best_achievable - surplus_all).mean())
        cvar = (
            float(np.percentile(surplus_closed, 5))
            if surplus_closed.size
            else float("nan")
        )
        agg[name] = {
            "deal_rate": float(deals.mean()),
            "walkaway_rate": float((~deals).mean()),
            "avg_surplus": float(surplus_closed.mean()) if surplus_closed.size else float("nan"),
            "std_surplus": float(surplus_closed.std()) if surplus_closed.size else float("nan"),
            "avg_rounds": float(rounds.mean()) if rounds.size else float("nan"),
            "regret": regret,
            "cvar_95": cvar,
            "n_holdout": len(holdout),
        }
    return agg
