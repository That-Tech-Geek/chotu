"""Atomic once-only execution with unknown-outcome reconciliation.

Claim-then-execute against ONE authoritative primitive (Upstash Redis
SET NX). Supabase is a durable ledger, never a second arbiter — we only
consult it when the lock primitive is *unavailable*, never when the lock
says 'already_claimed'.

Outcome taxonomy (never conflate them):
    SUCCESS   — provider confirmed the side effect.
    DEFINITE FAILURE — provider responded with a refusal (4xx/5xx body).
    UNKNOWN   — timeout/no response. The effect may have happened. We hold
                the claim and require reconciliation before any retry.
"""
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

    def _claim(self, key: str, payload: dict):
        """ONE arbiter: the lock. Supabase is consulted ONLY when the lock
        is unavailable — never when the lock authoritatively denies."""
        claim = self.lock.claim(key, payload)
        if claim.claimed:
            return claim
        if claim.reason == "already_claimed":
            # Authoritative denial from the single arbiter. Stop here.
            return claim
        # Lock unavailable/unconfigured — ledger as independent fallback.
        if self.ledger.live:
            return self.ledger.claim(key, payload)
        return claim

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
        claim = self._claim(key, payload)
        if not claim.claimed:
            return ExecutionResult(
                ActionState.BLOCKED,
                reason=f"duplicate_claim:{claim.reason}",
            )
        if shadow:
            self.lock.mark_executed(key, {**payload, "result": "shadow"})
            return ExecutionResult(ActionState.EXECUTED, reason="shadow_mode")
        try:
            resp = self.connector.send(deal)
        except Exception as e:  # connector violated its contract and raised
            self.lock.mark_unknown(key, payload)
            return ExecutionResult(ActionState.UNKNOWN, reason=str(e))
        if resp.get("success"):
            self.lock.mark_executed(key, {**payload, "response": resp})
            return ExecutionResult(ActionState.EXECUTED, provider_response=resp)
        if resp.get("outcome") == "unknown":
            # Timeout/no-response: the side effect may have happened. HOLD
            # the claim — reconciliation must decide, not a blind retry.
            self.lock.mark_unknown(key, {**payload, "provider_error": resp.get("error")})
            return ExecutionResult(
                ActionState.UNKNOWN,
                reason=resp.get("error", "provider did not respond"),
            )
        # Definite refusal — release our claim so a retry with a fixed deal
        # (new fingerprint) or an admin decision can proceed.
        self.lock.release(key, payload)
        return ExecutionResult(ActionState.FAILED, reason=resp.get("error", "unknown"))

    def reconcile(
        self,
        negotiation_id: str,
        action_id: str,
        sequence: int,
        provider_confirmed: bool,
        provider_reference: str | None = None,
    ) -> ExecutionResult:
        """Resolve an UNKNOWN action. Called by an out-of-band reconciler
        (cron or manual) after querying the provider's own state — e.g.
        Razorpay GET /payments?reference_id=... or Shopify order lookup.

        provider_confirmed=True  -> mark EXECUTED (provider reference stored)
        provider_confirmed=False -> CAS-release the claim so a retry is legal
        """
        key = f"{negotiation_id}:{action_id}:{sequence}"
        owner = self.lock._get(key)
        if owner is None:
            return ExecutionResult(ActionState.BLOCKED, reason="no_claim_to_reconcile")
        if provider_confirmed:
            self.lock.mark_executed(
                key, {**owner, "provider_reference": provider_reference},
            )
            return ExecutionResult(ActionState.EXECUTED,
                                   reason="reconciled_by_provider",
                                   provider_response={"reference": provider_reference})
        self.lock.release(key, owner)
        return ExecutionResult(ActionState.RETRYABLE, reason="provider_confirms_no_effect")
