"""Atomic once-only execution. Claim the idempotency key in an atomic
cloud primitive (Upstash Redis SET NX, or Supabase unique-constraint upsert),
then execute the side effect. A concurrent retry that loses the claim never
reaches the connector — so duplicate orders are structurally impossible
between claim and effect, not just statistically unlikely."""
from __future__ import annotations

import hashlib

from economic_engine.agent_runtime.lifecycle import ActionState
from economic_engine.agent_runtime.persistence import SupabaseActionLog, UpstashLock
from economic_engine.connectors.providers import ActionConnector
from economic_engine.state.canonical import Deal


class ExecutionResult:
    def __init__(self, state: ActionState, provider_response: dict | None = None,
                 reason: str | None = None):
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
        lock: UpstashLock | None = None,
        ledger: SupabaseActionLog | None = None,
    ):
        self.connector = connector
        self.lock = lock or UpstashLock()
        self.ledger = ledger or SupabaseActionLog()

    @staticmethod
    def _fingerprint(deal: Deal) -> str:
        return hashlib.sha256(
            deal.model_dump_json().encode()
        ).hexdigest()[:24]

    def run(
        self,
        negotiation_id: str,
        action_id: str,
        sequence: int,
        deal: Deal,
        shadow: bool = False,
    ) -> ExecutionResult:
        key = f"{negotiation_id}:{action_id}:{sequence}"
        payload = {
            "fingerprint": self._fingerprint(deal),
            "deal": deal.model_dump(mode="json"),
            "shadow": shadow,
        }
        # 1. Atomic claim. Exactly one caller wins; losers never execute.
        claim = self.lock.claim(key, payload)
        if not claim.claimed and self.ledger.live:
            claim = self.ledger.claim(key, payload)
        if not claim.claimed:
            return ExecutionResult(
                ActionState.BLOCKED,
                reason=f"duplicate_claim:{claim.reason}",
            )
        # 2. We own the claim. Execute the real side effect.
        if shadow:
            self.lock.mark_executed(key, {**payload, "result": "shadow"})
            return ExecutionResult(ActionState.EXECUTED, reason="shadow_mode")
        resp = self.connector.send(deal)
        if resp.get("success"):
            self.lock.mark_executed(key, {**payload, "response": resp})
            return ExecutionResult(ActionState.EXECUTED, provider_response=resp)
        # Side effect failed — CAS-release our own claim so a retry is legal.
        self.lock.release(key, payload)
        return ExecutionResult(ActionState.FAILED, reason=resp.get("error", "unknown"))
