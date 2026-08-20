"""Deterministic state machine. Transitions are exhaustively enumerated;
no FALLTHROUGH to hallucinated states. LLM proposals are validated against
this before execution."""
from __future__ import annotations

import enum


class NegotiationState(enum.Enum):
    INIT = "INIT"
    QUOTE_RECEIVED = "QUOTE_RECEIVED"
    OFFER_GENERATED = "OFFER_GENERATED"
    OFFER_SENT = "OFFER_SENT"
    COUNTER_RECEIVED = "COUNTER_RECEIVED"
    EVALUATING = "EVALUATING"
    ACCEPTED = "ACCEPTED"
    COUNTERED = "COUNTERED"
    WALKAWAY = "WALKAWAY"
    ESCALATE = "ESCALATE"
    TERMINAL = "TERMINAL"


TRANSITIONS: dict[NegotiationState, set[NegotiationState]] = {
    NegotiationState.INIT: {NegotiationState.QUOTE_RECEIVED},
    NegotiationState.QUOTE_RECEIVED: {NegotiationState.OFFER_GENERATED},
    NegotiationState.OFFER_GENERATED: {NegotiationState.OFFER_SENT},
    NegotiationState.OFFER_SENT: {NegotiationState.COUNTER_RECEIVED},
    NegotiationState.COUNTER_RECEIVED: {NegotiationState.EVALUATING},
    NegotiationState.EVALUATING: {
        NegotiationState.ACCEPTED,
        NegotiationState.COUNTERED,
        NegotiationState.WALKAWAY,
        NegotiationState.ESCALATE,
        NegotiationState.TERMINAL,
    },
    NegotiationState.ACCEPTED: {NegotiationState.TERMINAL},
    NegotiationState.COUNTERED: {NegotiationState.WALKAWAY,
        NegotiationState.ESCALATE, NegotiationState.TERMINAL},
    NegotiationState.WALKAWAY: {NegotiationState.TERMINAL},
    NegotiationState.ESCALATE: {NegotiationState.TERMINAL},
    NegotiationState.TERMINAL: set(),
}


class NegotiationStateMachine:
    def __init__(self):
        self.state = NegotiationState.INIT

    def transition(self, to: NegotiationState) -> NegotiationState:
        if to not in TRANSITIONS[self.state]:
            raise ValueError(
                f"illegal transition {self.state} -> {to}"
            )
        self.state = to
        return self.state
