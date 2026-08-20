"""Real execution against an ActionConnector, with durable idempotency and
the full ActionState lifecycle."""
from __future__ import annotations

from economic_engine.agent_runtime.lifecycle import ActionState
from economic_engine.agent_runtime.persistence import IdempotencyStore
from economic_engine.connectors.providers import ActionConnector
from economic_engine.state.canonical import Deal


class ExecutionResult:
    def __init__(self, state: ActionState, provider_response: dict | None = None, reason: str | None = None):
        self.state = state
        self.provider_response = provider_response
        self.reason = reason

    def __repr__(self) -> str:
        return (
            f"ExecutionResult({self.state.value}"
            + (f", reason={self.reason})" if self.reason else ")")
        )


class Executor:
    def __init__(
        self,
        connector: ActionConnector,
        idempotency: IdempotencyStore | None = None,
    ):
        self.connector = connector
        self.store = idempotency or IdempotencyStore()

    def run(
        self,
        negotiation_id: str,
        action_id: str,
        sequence: int,
        deal: Deal,
        shadow: bool = False,
    ) -> ExecutionResult:
        key = self.store.cache_key(negotiation_id, action_id, sequence)
        if self.store.seen(key):
            return ExecutionResult(ActionState.BLOCKED, reason="duplicate")
        # Validated
        # Queued
        # Executing
        # Shadow never sends — but still records.
        if shadow:
            self.store.record(key, {"shadow": True, "payload": deal.model_dump()})
            return ExecutionResult(ActionState.EXECUTED, reason="shadow_mode")
        # Actual side effect.
        resp = self.connector.send(deal)
        if resp.get("success"):
            self.store.record(key, {"response": resp, "payload": deal.model_dump()})
            return ExecutionResult(ActionState.EXECUTED, provider_response=resp)
        self.store.record(key, {"error": resp.get("error"), "payload": deal.model_dump()})
        return ExecutionResult(ActionState.FAILED, reason=resp.get("error", "unknown"))
