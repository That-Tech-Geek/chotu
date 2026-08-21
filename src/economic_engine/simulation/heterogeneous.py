"""Heterogeneous supplier population + within-supplier counterfactual.

Seven archetypes with genuinely different latent behaviour. Chotu gets NO
prior knowledge of which type it faces. Every supplier instance is played
against Fixed, Nash, and the Ensemble with identical initial conditions;
the report is the per-supplier ΔEV distribution (EV_chotu − EV_fixed).
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


class ArchetypedOpponent(SyntheticOpponent):
    """SyntheticOpponent extended with archetype-specific response shaping."""

    def __init__(self, archetype: str, **kwargs):
        super().__init__(**kwargs)
        self.archetype = archetype

    def respond(self, offer_price: float) -> str:
        if self.archetype == "aggressive_anchor_weak_reservation":
            # Talks big but folds fast: counter-offers stay high, yet its
            # effective acceptance threshold is well below the ask.
            if offer_price >= self.reservation * 1.02:
                return "ACCEPT"
            if self.round > 2 and offer_price >= self.reservation * 0.95:
                return "ACCEPT"
            if offer_price < self.reservation * 0.8 and self.round > 3:
                return "WALKAWAY"
            return "COUNTER"
        if self.archetype == "relationship_sensitive":
            # Concedes more when it has been dealt with repeatedly (here:
            # concessions accelerate with round count as trust proxy).
            self.patience = min(self.patience + 0.02 * self.round, 0.95)
            return super().respond(offer_price)
        if self.archetype == "quantity_sensitive":
            # Accepts noticeably below reservation if the implied volume is
            # large (we proxy volume pressure via the offer's generosity).
            if offer_price >= self.reservation * 0.96:
                return "ACCEPT"
            return super().respond(offer_price)
        if self.archetype == "delivery_sensitive":
            # Hardened on price; behaves like high_reservation but with a
            # tighter walkaway trigger on under-reservation offers.
            if offer_price < self.reservation * 0.95 and self.round > 2:
                return "WALKAWAY"
            return super().respond(offer_price)
        return super().respond(offer_price)


def make_archetype(archetype: str, base: float, rng_seed: int) -> ArchetypedOpponent:
    rng = np.random.default_rng(rng_seed)
    if archetype == "A_low_res_high_patience":
        res = base * float(rng.uniform(0.70, 0.80))
        return ArchetypedOpponent(archetype, supply_cost=res * 0.75,
                                  reservation=res, patience=0.9,
                                  urgency=0.1, rng=np.random.default_rng(rng_seed + 1))
    if archetype == "B_high_res_low_patience":
        res = base * float(rng.uniform(0.88, 0.97))
        return ArchetypedOpponent(archetype, supply_cost=res * 0.75,
                                  reservation=res, patience=0.2,
                                  urgency=0.3, rng=np.random.default_rng(rng_seed + 1))
    if archetype == "C_aggressive_anchor_weak_res":
        res = base * float(rng.uniform(0.72, 0.82))
        return ArchetypedOpponent(archetype, supply_cost=res * 0.75,
                                  reservation=res, patience=0.4,
                                  urgency=0.4, ask_markup=0.45,
                                  rng=np.random.default_rng(rng_seed + 1))
    if archetype == "D_relationship_sensitive":
        res = base * float(rng.uniform(0.78, 0.88))
        return ArchetypedOpponent(archetype, supply_cost=res * 0.75,
                                  reservation=res, patience=0.3,
                                  urgency=0.3, rng=np.random.default_rng(rng_seed + 1))
    if archetype == "E_high_urgency":
        res = base * float(rng.uniform(0.75, 0.90))
        return ArchetypedOpponent(archetype, supply_cost=res * 0.75,
                                  reservation=res, patience=0.4,
                                  urgency=1.0, rng=np.random.default_rng(rng_seed + 1))
    if archetype == "F_quantity_sensitive":
        res = base * float(rng.uniform(0.78, 0.90))
        return ArchetypedOpponent(archetype, supply_cost=res * 0.75,
                                  reservation=res, patience=0.5,
                                  urgency=0.3, rng=np.random.default_rng(rng_seed + 1))
    # G_delivery_sensitive
    res = base * float(rng.uniform(0.85, 0.95))
    return ArchetypedOpponent(archetype, supply_cost=res * 0.75,
                              reservation=res, patience=0.4,
                              urgency=0.2, rng=np.random.default_rng(rng_seed + 1))


ARCHETYPES = [
    "A_low_res_high_patience",
    "B_high_res_low_patience",
    "C_aggressive_anchor_weak_res",
    "D_relationship_sensitive",
    "E_high_urgency",
    "F_quantity_sensitive",
    "G_delivery_sensitive",
]


def run_heterogeneous_experiment(
    per_archetype: int = 12,
    c_walk_ratio: float = 0.25,
    seed: int = 20000,
) -> dict:
    policies = {
        "fixed": lambda: _PolicyAdapter(BaselinePolicyFactory.fixed_price()),
        "nash": lambda: _PolicyAdapter(BaselinePolicyFactory.nash()),
        "chotu": lambda: EnsemblePolicy(),
    }
    rows = []
    for a_idx, archetype in enumerate(ARCHETYPES):
        for j in range(per_archetype):
            seed_i = seed + a_idx * 1000 + j
            base = float(np.random.default_rng(seed_i).uniform(80, 120))
            c_walk = c_walk_ratio * base
            for name, factory in policies.items():
                ctx = make_ctx(base=base, seed=seed_i)
                opp = make_archetype(archetype, base, rng_seed=seed_i + 7)
                r = run_negotiation(factory(), ctx, opp, base)
                walked = 0.0 if r["accepted"] else 1.0
                rows.append({
                    "archetype": archetype,
                    "supplier_seed": seed_i,
                    "policy": name,
                    "ev": r["surplus"] - c_walk * walked,
                    "surplus": r["surplus"],
                    "walked": walked,
                })
    return _report(rows, per_archetype)


def _report(rows: list, per_archetype: int) -> dict:
    by_policy: dict[str, list[float]] = {}
    by_archetype: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row["ev"])
        by_archetype.setdefault(row["archetype"], {}).setdefault(
            row["policy"], [],
        ).append(row["ev"])
    # Within-supplier counterfactual: for each archetype, mean(EV_chotu) −
    # mean(EV_fixed) over identical supplier instances.
    delta = {}
    for archetype, evs in by_archetype.items():
        delta[archetype] = {
            "chotu_ev": float(np.mean(evs.get("chotu", [float("nan")]))),
            "fixed_ev": float(np.mean(evs.get("fixed", [float("nan")]))),
            "nash_ev": float(np.mean(evs.get("nash", [float("nan")]))),
            "delta_chotu_minus_fixed": float(
                np.mean(evs.get("chotu", [0])) - np.mean(evs.get("fixed", [0]))
            ),
            "delta_chotu_minus_nash": float(
                np.mean(evs.get("chotu", [0])) - np.mean(evs.get("nash", [0]))
            ),
        }
    overall = {
        name: {
            "ev": float(np.mean(evs)),
            "ev_se": float(np.std(evs, ddof=1) / np.sqrt(len(evs))),
        }
        for name, evs in by_policy.items()
    }
    # Regime router: pick the empirically better policy per archetype.
    router_ev = float(np.mean([
        max(v["chotu_ev"], v["fixed_ev"]) for v in delta.values()
    ]))
    wins = sum(1 for v in delta.values() if v["delta_chotu_minus_fixed"] > 0.05)
    losses = sum(1 for v in delta.values() if v["delta_chotu_minus_fixed"] < -0.05)
    return {
        "overall": overall,
        "by_archetype": delta,
        "n_per_cell": per_archetype,
        "router_ev": router_ev,
        "outcomes": {"chotu_wins": wins, "fixed_wins": losses,
                     "ties": len(delta) - wins - losses},
    }
