"""Autonomous runtime: envelope-check -> kill-switch -> state transition ->
idempotent execute -> audit. The economic engine returns a Decision; this
body validates it and executes; the brain never directly sends."""
from __future__ import annotations

from economic_engine.agent_runtime.envelope import PolicyEnvelope
from economic_engine.agent_runtime.idempotency import ActionExecutor
from economic_engine.agent_runtime.kill_switch import KillSwitch
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
        audit_callback=None,
    ):
        self.envelope = envelope
        self.kill = kill
        self.executor = ActionExecutor()
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
        violations = self.envelope.check_deal(deal)
        stopped = self.kill.evaluate(deal, ctx)
        if violations:
            self.audit("envelope_violation", violations)
            return {"action": "BLOCKED", "violations": violations}
        if stopped:
            self.audit("kill_switch", stopped)
            return {"action": "BLOCKED", "reason": stopped}
        try:
            # Every EVALUATING response must follow a received counter —
            # the brain can propose, the state machine decides.
            if self.sm.state != NegotiationState.TERMINAL:
                try:
                    self.sm.transition(
                        NegotiationState.EVALUATING
                        if self.sm.state != NegotiationState.INIT
                        else NegotiationState.QUOTE_RECEIVED
                    )
                except ValueError:
                    self.audit("illegal_transition", self.sm.state.value)
                    return {
                        "action": "BLOCKED",
                        "reason": f"illegal transition from {self.sm.state.value}",
                    }
        except Exception:
            pass
        if shadow:
            self.audit("shadow_decision", deal.model_dump())
            return {
                "action": "SHADOW",
                "deal": deal.model_dump(),
                "executed": False,
            }
        out = self.executor.execute(negotiation_id, action_id, sequence, deal)
        self.audit("execute", out)
        return out
