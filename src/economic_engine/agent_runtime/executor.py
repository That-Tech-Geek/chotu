"""Autonomous runtime: envelope-check -> kill-switch -> state transition ->
idempotent execute -> audit. The economic engine returns a Decision; this
body validates it and executes; the brain never directly sends."""
from __future__ import annotations

from economic_engine.agent_runtime.envelope import PolicyEnvelope
from economic_engine.agent_runtime.execution import Executor
from economic_engine.agent_runtime.kill_switch import KillSwitch
from economic_engine.agent_runtime.lifecycle import ActionState
from economic_engine.agent_runtime.state_machine import (
    NegotiationState,
    NegotiationStateMachine,
)
from economic_engine.state.canonical import Deal, NegotiationContext


class AutonomousRuntime:
    def __init__(
        self,
        envelope: PolicyEnvelope,
        kill: KillSwitch,
        executor: Executor,
        audit_callback=None,
    ):
        self.envelope = envelope
        self.kill = kill
        self.executor = executor
        self.sm = NegotiationStateMachine()
        self.audit = audit_callback or (lambda event, data: None)

    def handle(
        self,
        negotiation_id: str,
        action_id: str,
        sequence: int,
        deal: Deal,
        ctx: NegotiationContext,
        shadow: bool = False,
    ) -> dict:
        # Envelope first — brain cannot out-vote a hard boundary.
        base_cost = ctx.product.base_purchase_cost if ctx.product else None
        violations = self.envelope.check_deal(deal, base_cost=base_cost)
        if violations:
            self.audit("envelope_violation", violations)
            return {"action": "BLOCKED", "violations": violations}
        stopped = self.kill.evaluate(deal, ctx)
        if stopped:
            self.audit("kill_switch", stopped)
            return {"action": "BLOCKED", "reason": stopped}
        # Advance the state machine deterministically through each
        # well-formed hop.
        transition_map = {
            NegotiationState.INIT: NegotiationState.QUOTE_RECEIVED,
            NegotiationState.QUOTE_RECEIVED: NegotiationState.OFFER_GENERATED,
            NegotiationState.OFFER_GENERATED: NegotiationState.OFFER_SENT,
            NegotiationState.OFFER_SENT: NegotiationState.COUNTER_RECEIVED,
            NegotiationState.COUNTER_RECEIVED: NegotiationState.EVALUATING,
        }
        target = transition_map.get(self.sm.state)
        if target is not None:
            try:
                self.sm.transition(target)
            except ValueError as e:
                self.audit("illegal_transition", str(e))
                return {"action": "BLOCKED", "reason": str(e)}
        # Execute side effect.
        result = self.executor.run(negotiation_id, action_id, sequence, deal, shadow=shadow)
        if result.state == ActionState.EXECUTED:
            self.audit("execute", {"deal": deal.model_dump(), "shadow": shadow})
            # Post-execution state; for now we stay in EVALUATING until a
            # response arrives (counter-receipt flows back through events).
            return {
                "action": "EXECUTED",
                "state": result.state.value,
                "shadow": shadow,
            }
        if result.state == ActionState.FAILED:
            return {"action": "FAILED", "reason": result.reason}
        return {"action": result.state.value}
