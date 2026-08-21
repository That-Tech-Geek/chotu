"""Shadow mode: Chotu proposes, runtime validates, DO NOT SEND, record
hypothetical outcome vs baseline actual outcome."""
from __future__ import annotations

import numpy as np

from economic_engine.agent_runtime.envelope import PolicyEnvelope
from economic_engine.agent_runtime.execution import Executor
from economic_engine.agent_runtime.executor import AutonomousRuntime
from economic_engine.agent_runtime.kill_switch import KillSwitch
from economic_engine.simulation.benchmark import (
    ChotuPolicy,
    make_ctx,
    run_negotiation,
)
from economic_engine.simulation.opponent import SyntheticOpponent
from economic_engine.state.canonical import Deal


class ShadowPolicy:
    """Chotu wrapped in AutonomousRuntime with shadow=True."""

    def __init__(self):
        from economic_engine.connectors.providers import MockConnector

        envelope = PolicyEnvelope(
            max_unit_price=200.0,
            min_unit_price=0.0,
            max_total_spend=1_000_000.0,
        )
        kill = KillSwitch(max_price=200.0)
        self.runtime = AutonomousRuntime(
            envelope, kill, Executor(connector=MockConnector()),
        )
        self.brain = ChotuPolicy()

    def next_offer(self, ctx, current_offer):
        price = self.brain.next_offer(ctx, current_offer)
        if price is None:
            return None
        deal = Deal(price=price, quantity=ctx.negotiation.quantity, currency="INR")
        out = self.runtime.handle(
            negotiation_id=ctx.negotiation.id,
            action_id=f"a_{len(ctx.negotiation.rounds)}",
            sequence=len(ctx.negotiation.rounds),
            deal=deal,
            ctx=ctx,
            shadow=True,
        )
        if out["action"] == "BLOCKED":
            return None
        return float(price)


def run_shadow_holdout(n_holdout: int = 30, seed: int = 3000) -> dict:
    policy = ShadowPolicy()
    surplus, deals, hyp = [], [], []
    for i in range(n_holdout):
        seed_i = seed + i
        base = float(np.random.default_rng(seed_i).uniform(80, 120))
        supplier_reservation = float(
            np.random.default_rng(seed_i + 1).uniform(0.75, 0.95) * base,
        )
        ctx = make_ctx(base=base, seed=seed_i)
        opp = SyntheticOpponent(
            supply_cost=supplier_reservation * 0.75,
            reservation=supplier_reservation,
            patience=float(np.random.default_rng(seed_i + 2).uniform(0.3, 0.9)),
            urgency=float(np.random.default_rng(seed_i + 3).uniform(0.0, 1.0)),
            rng=np.random.default_rng(seed_i + 4),
        )
        r = run_negotiation(policy, ctx, opp, base)
        surplus.append(r["surplus"])
        deals.append(r["accepted"])
        hyp.append(r["rounds"])
    return {
        "deal_rate": float(np.mean(deals)),
        "avg_surplus": float(np.mean(surplus)),
        "avg_rounds": float(np.mean(hyp)),
    }
