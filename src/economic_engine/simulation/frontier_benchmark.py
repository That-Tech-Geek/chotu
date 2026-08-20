"""Chotu Pareto frontier vs union of all baseline frontiers (each baseline
sweeps its control knob step-wise). Answers 'is Chotu on the overall efficient
frontier?' instead of comparing it to one baseline lambda."""
from __future__ import annotations

import numpy as np

from economic_engine.simulation.benchmark import (
    BaselinePolicyFactory,
    ChotuPolicy,
    _PolicyAdapter,
    make_ctx,
    run_negotiation,
)
from economic_engine.simulation.opponent import SyntheticOpponent
from economic_engine.simulation.pareto import sweep_lambdas


def _evaluate_policy(policy_fn, opponents: list[dict]) -> tuple:
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
        r = run_negotiation(policy_fn, ctx, opp, opp_cfg["base"])
        surplus.append(r["surplus"])
        deals.append(r["accepted"])
    return float(np.mean(deals)), float(np.mean(surplus))


def _frontier(points: list[tuple]) -> list[tuple]:
    pts = sorted(points, key=lambda p: (-p[0], p[1]))
    frontier = []
    best_surplus = -float("inf")
    for p in pts:
        if p[1] > best_surplus:
            frontier.append(p)
            best_surplus = p[1]
    return frontier


def _sweep_baseline(
    factory_fn,
    knob_name: str,
    knob_values: list[float],
    opponents: list[dict],
) -> list[tuple]:
    pts = []
    for val in knob_values:
        policy_fn = _PolicyAdapter(factory_fn(val))
        deal, surplus = _evaluate_policy(policy_fn, opponents)
        pts.append((deal, surplus))
    return pts


def run_frontier_benchmark(n: int = 30, seed: int = 800) -> dict:
    from economic_engine.simulation.pareto import _make_opponents
    opponents = _make_opponents(seed, n)
    all_points = []
    chotu_points = []
    for name, (factory, values) in _baseline_configs().items():
        pts = _sweep_baseline(factory, name, values, opponents)
        frontier_pts = _frontier(pts)
        for p in frontier_pts:
            all_points.append({"policy": name, "deal": p[0], "surplus": p[1]})
    for row in sweep_lambdas(n_eval=n, seed=seed + 100):
        chotu_points.append({"policy": "chotu", "deal": row["deal_rate"],
                            "surplus": row["avg_surplus"]})
    chotu_frontier = _frontier([(p["deal"], p["surplus"]) for p in chotu_points])
    combined = all_points + chotu_points
    combined_sorted = sorted(
        combined, key=lambda p: (-p["deal"], p["surplus"]),
    )
    best_surplus = -float("inf")
    global_frontier = []
    for p in combined_sorted:
        if p["surplus"] > best_surplus:
            global_frontier.append(p)
            best_surplus = p["surplus"]
    return {
        "chotu_frontier_size": len(chotu_frontier),
        "chotu_points_on_global_frontier": sum(
            1 for p in global_frontier if p["policy"] == "chotu"
        ),
        "global_frontier": global_frontier,
        "baseline_points_count": len(all_points),
    }


def _baseline_configs() -> dict:
    return {
        "concession": (BaselinePolicyFactory.concession, [0.02, 0.05, 0.1]),
        "tft": (BaselinePolicyFactory.tit_for_tat, [0.85, 0.93, 0.98]),
        "nash": (BaselinePolicyFactory.nash, []),  # static
        "random": (
            lambda seed: BaselinePolicyFactory.random(seed=seed),
            [0, 100, 1000],
        ),
    }
