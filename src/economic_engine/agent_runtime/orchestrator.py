"""Orchestrator: supplier event -> decide -> envelope -> kill -> state ->
idempotent execute -> audit. The full closed loop."""
from __future__ import annotations

import uuid

from economic_engine.agent_runtime.envelope import PolicyEnvelope
from economic_engine.agent_runtime.execution import Executor
from economic_engine.agent_runtime.kill_switch import KillSwitch
from economic_engine.agent_runtime.executor import AutonomousRuntime
from economic_engine.negotiation.engine import NegotiationEngine
from economic_engine.state.canonical import (
    Deal,
    NegotiationContext,
    Offer,
    Round,
)


class Orchestrator:
    def __init__(
        self,
        engine: NegotiationEngine,
        envelope: PolicyEnvelope,
        kill: KillSwitch,
        executor: Executor,
    ):
        self.engine = engine
        self.runtime = AutonomousRuntime(envelope, kill, executor)

    def handle_event(
        self,
        ctx: NegotiationContext,
        supplier_message: str,
        price: float | None,
        shadow: bool = False,
    ) -> dict:
        if price is not None:
            ctx.negotiation.rounds.append(
                Round(
                    index=len(ctx.negotiation.rounds),
                    offer=Offer(price=price, actor="supplier"),
                )
            )
        decision = self.engine.decide(ctx)
        deal = Deal(
            price=decision.get("price"),
            quantity=ctx.negotiation.quantity,
            currency="INR",
        )
        return self.runtime.handle(
            negotiation_id=ctx.negotiation.id,
            action_id=str(uuid.uuid4()),
            sequence=len(ctx.negotiation.rounds),
            deal=deal,
            ctx=ctx,
            shadow=shadow,
        )
