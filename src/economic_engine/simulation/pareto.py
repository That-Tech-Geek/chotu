"""Risk-aversion sweep over lambda to map the (deal_rate, surplus) Pareto
frontier for Chotu vs baselines. Answers 'is the engine on the efficient
frontier?' rather than 'does it win on one metric?'.
"""
from __future__ import annotations

import numpy as np

from economic_engine.negotiation.engine import NegotiationEngine
from economic_engine.simulation.benchmark import (
    BaselinePolicyFactory,
    ChotuPolicy,
    _PolicyAdapter,
    make_ctx,
    run_negotiation,
)
from economic_engine.simulation.opponent import SyntheticOpponent


def _make_opponents(seed_start: int, n: int) -> list[dict]:
    out = []
    for i in range(n):
        seed_i = seed_start + i
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


def sweep_lambdas(
    lambdas: list[float] | None = None,
    n_eval: int = 50,
    seed: int = 100,
) -> list[dict]:
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    opponents = _make_opponents(seed, n_eval)
    rows = []
    for lam in lambdas:
        engine = NegotiationEngine(mc_mode="FAST", lambda_override=lam)
        surplus, deals, regrets = [], [], []
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
            policy.engine = engine
            r = run_negotiation(policy, ctx, opp, opp_cfg["base"])
            surplus.append(r["surplus"])
            deals.append(r["accepted"])
            regrets.append(max(r["surplus"] * 0.95, 0.0) - r["surplus"])
        rows.append({
            "lambda": lam,
            "deal_rate": float(np.mean(deals)),
            "avg_surplus": float(np.mean(surplus)),
            "regret": float(np.mean(regrets)),
        })
    return rows


def pareto_rows(rows: list[dict]) -> list[dict]:
    """Filter to the efficient frontier (deal_rate non-decreasing, surplus
    non-decreasing)."""
    sorted_rows = sorted(rows, key=lambda r: (-r["deal_rate"], r["avg_surplus"]))
    frontier = []
    best_surplus = -float("inf")
    for r in sorted_rows:
        if r["avg_surplus"] > best_surplus:
            frontier.append(r)
            best_surplus = r["avg_surplus"]
    return frontier


def baseline_points(n_eval: int = 50, seed: int = 200) -> dict:
    opponents = _make_opponents(seed, n_eval)
    baselines = {
        "fixed": BaselinePolicyFactory.fixed_price,
        "concession": BaselinePolicyFactory.concession,
        "tft": BaselinePolicyFactory.tit_for_tat,
        "random": BaselinePolicyFactory.random,
        "nash": BaselinePolicyFactory.nash,
    }
    out = {}
    for name, factory in baselines.items():
        policy_fn = factory()
        surplus, deals = [], []
        for opp_cfg in opponents:
            ctx = make_ctx(base=opp_cfg["base"], seed=opp_cfg["seed"])
            opp = SyntheticOpponent(
                supply_cost=opp_cfg["supply_cost"],
                reservation=opp_cfg["supplier_reservation"],
                patience=opp_cfg["patience"],
                urgency=opp_cfg["urgency"],
                rng=np.random.default_rng(opp_cfg["seed"] + 4),
            )
            r = run_negotiation(_PolicyAdapter(policy_fn), ctx, opp, opp_cfg["base"])
            surplus.append(r["surplus"])
            deals.append(r["accepted"])
        out[name] = {
            "deal_rate": float(np.mean(deals)),
            "avg_surplus": float(np.mean(surplus)),
        }
    return out
