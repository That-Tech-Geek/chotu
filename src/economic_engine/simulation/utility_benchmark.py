"""The merchant's utility: EV = Surplus − C_walk · P(walkaway).

No new models. One evaluation across all policies on identical opponents,
with confidence intervals, answering: does Chotu beat Nash at the merchant's
actual utility function?
"""
from __future__ import annotations

import numpy as np

from economic_engine.simulation.benchmark import (
    BaselinePolicyFactory,
    ChotuPolicy,
    _PolicyAdapter,
    make_ctx,
    run_negotiation,
)
from economic_engine.simulation.meta import EnsemblePolicy
from economic_engine.simulation.opponent import SyntheticOpponent


def _make_opponent(seed_i: int, base: float):
    sr = float(np.random.default_rng(seed_i + 1).uniform(0.75, 0.95) * base)
    return SyntheticOpponent(
        supply_cost=sr * 0.75,
        reservation=sr,
        patience=float(np.random.default_rng(seed_i + 2).uniform(0.3, 0.9)),
        urgency=float(np.random.default_rng(seed_i + 3).uniform(0.0, 1.0)),
        rng=np.random.default_rng(seed_i + 4),
    ), sr


def run_utility_benchmark(
    n: int = 200,
    c_walk_ratio: float = 0.25,
    seed: int = 7000,
) -> dict:
    """c_walk_ratio scales the walkaway cost against base price: a walkaway
    costs the merchant c_walk_ratio * base (sourcing delay, lost stock,
    expediting, alternative-supplier premium)."""
    policies = {
        "ensemble": lambda: EnsemblePolicy(),
        "chotu": lambda: ChotuPolicy(),
        "nash": lambda: _PolicyAdapter(BaselinePolicyFactory.nash()),
        "fixed": lambda: _PolicyAdapter(BaselinePolicyFactory.fixed_price()),
        "concession": lambda: _PolicyAdapter(BaselinePolicyFactory.concession()),
        "tft": lambda: _PolicyAdapter(BaselinePolicyFactory.tit_for_tat()),
        "random": lambda: _PolicyAdapter(BaselinePolicyFactory.random(seed=42)),
    }
    results = {name: {"utility": [], "surplus": [], "walked": []}
               for name in policies}
    for i in range(n):
        seed_i = seed + i
        base = float(np.random.default_rng(seed_i).uniform(80, 120))
        c_walk = c_walk_ratio * base
        for name, factory in policies.items():
            ctx = make_ctx(base=base, seed=seed_i)
            opp, _sr = _make_opponent(seed_i, base)
            r = run_negotiation(factory(), ctx, opp, base)
            walked = 0.0 if r["accepted"] else 1.0
            results[name]["utility"].append(r["surplus"] - c_walk * walked)
            results[name]["surplus"].append(r["surplus"])
            results[name]["walked"].append(walked)
    out = {}
    for name, r in results.items():
        u = np.asarray(r["utility"])
        out[name] = {
            "ev": float(u.mean()),
            "ev_se": float(u.std(ddof=1) / np.sqrt(len(u))),
            "deal_rate": float(1.0 - np.mean(r["walked"])),
            "walk_rate": float(np.mean(r["walked"])),
            "avg_surplus": float(np.mean(r["surplus"])),
        }
    out["_meta"] = {"n": n, "c_walk_ratio": c_walk_ratio, "seed": seed}
    return out


def run_cwalk_sweep(
    n: int = 100,
    c_walk_ratios: tuple = (0.0, 0.1, 0.25, 0.5, 1.0),
    seed: int = 9000,
) -> dict:
    """EV at multiple walkaway costs — the utility Pareto across the
    merchant's risk profile."""
    sweep = {}
    for ratio in c_walk_ratios:
        res = run_utility_benchmark(n=n, c_walk_ratio=ratio, seed=seed)
        sweep[ratio] = {k: v["ev"] for k, v in res.items() if k != "_meta"}
    return sweep
