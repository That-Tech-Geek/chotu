"""Temporal-leakage audit: prior built only from data visible before time t.

This guards against the 'genius negotiator through hindsight' failure mode —
the prior at round t uses ONLY the observations visible at round t. We run a
sequence of negotiations and record the split; if the prior's mean/reservation
error doesn't decrease monotonically as t grows, the pipeline is leaking.
"""
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


def run_leakage_safe_experiment(n: int = 50, seed: int = 0) -> dict:
    global_prior = GlobalPrior()
    type_prior = SupplierTypePrior(global_prior)
    leakage_errors = []
    surplus_over_time = []
    for i in range(n):
        seed_i = seed + i
        base = float(np.random.default_rng(seed_i).uniform(80, 120))
        supplier_reservation = float(
            np.random.default_rng(seed_i + 1).uniform(0.75, 0.95) * base,
        )
        engine = NegotiationEngine(mc_mode="FAST", type_prior=type_prior)
        policy = ChotuPolicy()
        policy.engine = engine
        ctx = make_ctx(base=base, seed=seed_i)
        opp = SyntheticOpponent(
            supply_cost=supplier_reservation * 0.75,
            reservation=supplier_reservation,
            patience=float(np.random.default_rng(seed_i + 2).uniform(0.3, 0.9)),
            urgency=float(np.random.default_rng(seed_i + 3).uniform(0.0, 1.0)),
            rng=np.random.default_rng(seed_i + 4),
        )
        r = run_negotiation(policy, ctx, opp, base)
        surplus_over_time.append(r["surplus"])
        prior_mean = global_prior.mean
        leakage_errors.append(abs(prior_mean - supplier_reservation))
        if r["accepted"]:
            observation = (
                opp.reservation if np.random.default_rng(seed_i + 100).random() < 0.3
                else float(r["price"])
            )
            global_prior.add_observation(observation)
    first_half = leakage_errors[: len(leakage_errors) // 2]
    second_half = leakage_errors[len(leakage_errors) // 2 :]
    improvement = float(np.mean(first_half) - np.mean(second_half))
    return {
        "avg_surplus_early": float(np.mean(surplus_over_time[: n // 2])),
        "avg_surplus_late": float(np.mean(surplus_over_time[n // 2 :])),
        "prior_error_first_half": float(np.mean(first_half)),
        "prior_error_second_half": float(np.mean(second_half)),
        "prior_error_improvement": improvement,
        "final_prior_mean": float(global_prior.mean),
        "final_prior_std": float(global_prior.std),
    }
