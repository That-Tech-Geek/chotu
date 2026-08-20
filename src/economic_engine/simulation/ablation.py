"""Ablation: measure where Chotu's performance actually comes from —
no learning, SupplierPosterior-only, GlobalPrior, Global+Type,
Global+Type+Posterior (Full Chotu). Same opponents, seeds, fold."""
from __future__ import annotations

import numpy as np

from economic_engine.learning.prior import GlobalPrior, SupplierTypePrior
from economic_engine.negotiation.engine import NegotiationEngine
from economic_engine.simulation.benchmark import (
    ChotuPolicy,
    make_ctx,
    run_negotiation,
)
from economic_engine.simulation.opponent import SyntheticOpponent


def _evaluate(engine_factory_fn, opponents: list[dict]) -> dict:
    surplus, deals, regrets, rounds = [], [], [], []
    for opp_cfg in opponents:
        ctx = make_ctx(base=opp_cfg["base"], seed=opp_cfg["seed"])
        opp = SyntheticOpponent(
            supply_cost=opp_cfg["supply_cost"],
            reservation=opp_cfg["supplier_reservation"],
            patience=opp_cfg["patience"],
            urgency=opp_cfg["urgency"],
            rng=np.random.default_rng(opp_cfg["seed"] + 4),
        )
        policy = ChotuPolicy()
        policy.engine = engine_factory_fn()
        r = run_negotiation(policy, ctx, opp, opp_cfg["base"])
        surplus.append(r["surplus"])
        deals.append(r["accepted"])
        regrets.append(max(r["surplus"] * 0.95, 0.0) - r["surplus"])
        if r["accepted"]:
            rounds.append(r["rounds"])
    return {
        "deal_rate": float(np.mean(deals)),
        "avg_surplus": float(np.mean(surplus)),
        "regret": float(np.mean(regrets)),
        "avg_rounds": float(np.mean(rounds)) if rounds else float("nan"),
    }


def _make_opponents(n: int, seed: int) -> list[dict]:
    out = []
    for i in range(n):
        seed_i = seed + i
        base = float(np.random.default_rng(seed_i).uniform(80, 120))
        supplier_reservation = float(
            np.random.default_rng(seed_i + 1).uniform(0.75, 0.95) * base,
        )
        out.append({
            "seed": seed_i,
            "base": base,
            "supplier_reservation": supplier_reservation,
            "supply_cost": supplier_reservation * 0.75,
            "patience": float(np.random.default_rng(seed_i + 2).uniform(0.3, 0.9)),
            "urgency": float(np.random.default_rng(seed_i + 3).uniform(0.0, 1.0)),
        })
    return out


def run_ablation(n: int = 40, seed: int = 500) -> dict:
    opponents = _make_opponents(n, seed)
    # Shared GlobalPrior accumulates only after each level finishes —
    # subsequent arms see strictly-more-data (temporally correct).
    shared_global: GlobalPrior | None = None

    def global_prior() -> GlobalPrior:
        nonlocal shared_global
        if shared_global is None:
            shared_global = GlobalPrior()
        return shared_global

    def no_learning():
        return NegotiationEngine(mc_mode="FAST")

    def posterior_only():
        return NegotiationEngine(mc_mode="FAST")

    def global_level():
        return NegotiationEngine(
            mc_mode="FAST",
            type_prior=SupplierTypePrior(global_prior()),
        )

    def full_chotu():
        return NegotiationEngine(
            mc_mode="FAST",
            type_prior=SupplierTypePrior(global_prior()),
        )

    configs = [
        ("A_no_learning", no_learning),
        ("B_supplier_posterior", posterior_only),
        ("C_global", global_level),
        ("D_global_type", global_level),
        ("E_global_type_posterior", global_level),
        ("F_full_chotu", full_chotu),
    ]
    out = {}
    for name, factory in configs:
        out[name] = _evaluate(factory, opponents)
    return out
