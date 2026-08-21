"""Atomic claim-then-execute idempotency. Two independent cloud-realtime
backends, no local caches:

1. Upstash Redis REST (`SET key val NX EX ttl`) — the atomic primitive.
   Redis `SET NX` is the CAS: exactly one caller wins the claim, so a
   retry/replay loses before the side effect ever fires. If the process dies
   between claim and execution, the lock TTL expires and the retry is safe
   to claim again (the side effect never ran).

2. Supabase `action_records` — upsert with `resolution=ignore-duplicates`,
   then `select` back. If the row was inserted by another request
   (409 / returned row's payload differs from ours), we treat it as claimed.

No file cache: reads/writes always hit the cloud store, so there is no
staleness window between two Vercel invocations."""
from __future__ import annotations

import os
import time
from typing import Any

import httpx


class ClaimResult:
    def __init__(self, claimed: bool, owner_payload: dict | None = None,
                 reason: str | None = None):
        self.claimed = claimed
        self.owner_payload = owner_payload
        self.reason = reason


class UpstashLock:
    """Atomic claim via Redis SET NX EX. Requires UPSTASH_REDIS_REST_URL
    and UPSTASH_REDIS_REST_TOKEN env vars."""

    def __init__(self, url: str | None = None, token: str | None = None,
                 ttl_seconds: int = 300, timeout: float = 3.0):
        self.url = (url if url is not None else os.environ.get("UPSTASH_REDIS_REST_URL", "")).rstrip("/")
        self.token = token if token is not None else os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
        self.ttl = ttl_seconds
        self.timeout = timeout

    @property
    def live(self) -> bool:
        return bool(self.url and self.token)

    def claim(self, key: str, payload: dict) -> ClaimResult:
        """SET key payload NX EX ttl — atomic. Returns claimed=True only for
        the single winner."""
        if not self.live:
            return ClaimResult(claimed=False, reason="upstash_not_configured")
        resp = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.token}"},
            json=["SET", key, _encode(payload), "NX", "EX", self.ttl],
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json().get("result")
        if result == "OK":
            return ClaimResult(claimed=True)
        # Key existed — fetch the existing owner payload for the audit trail.
        owner = self._get(key)
        return ClaimResult(claimed=False, owner_payload=owner, reason="already_claimed")

    def mark_executed(self, key: str, payload: dict) -> None:
        """Overwrite claim with terminal state (long TTL so late retries see it)."""
        if not self.live:
            return
        payload = {**payload, "executed": True, "executed_at": time.time()}
        httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.token}"},
            json=["SET", key, _encode(payload), "XX", "KEEPTTL"],
            timeout=self.timeout,
        ).raise_for_status()

    def release(self, key: str, payload: dict) -> None:
        """CAS-delete: only the claim owner (matching fingerprint) may release,
        so a slow/failed request can't erase a newer legitimate claim."""
        if not self.live:
            return
        owner = self._get(key)
        if not owner or owner.get("fingerprint") != payload.get("fingerprint"):
            return
        httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.token}"},
            json=["DEL", key],
            timeout=self.timeout,
        ).raise_for_status()

    def _get(self, key: str) -> dict | None:
        try:
            resp = httpx.post(
                self.url,
                headers={"Authorization": f"Bearer {self.token}"},
                json=["GET", key],
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("result")
            return _decode(raw) if raw else None
        except httpx.HTTPError:
            return None


class SupabaseActionLog:
    """Durable action ledger via Supabase REST. Upsert with ignore-duplicates:
    the DB's unique constraint on idempotency_key is the arbiter of who won.
    We only treat ourselves as the owner if the row we read back matches our
    payload fingerprint."""

    def __init__(self, url: str | None = None, key: str | None = None,
                 table: str = "action_records", timeout: float = 5.0):
        self.url = (url if url is not None else os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = key if key is not None else os.environ.get("SUPABASE_KEY", "")
        self.table = table
        self.timeout = timeout

    @property
    def live(self) -> bool:
        return bool(self.url and self.key)

    def _headers(self) -> dict:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    def claim(self, key: str, payload: dict) -> ClaimResult:
        if not self.live:
            return ClaimResult(claimed=False, reason="supabase_not_configured")
        fingerprint = payload.get("fingerprint")
        resp = httpx.post(
            f"{self.url}/rest/v1/{self.table}",
            headers={**self._headers(),
                     "Prefer": "resolution=ignore-duplicates,return=representation"},
            json={"idempotency_key": key, "payload": payload,
                  "fingerprint": fingerprint},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        rows = resp.json()
        if rows and rows[0].get("fingerprint") == fingerprint:
            return ClaimResult(claimed=True)
        return ClaimResult(claimed=False, reason="already_claimed")

    def lookup(self, key: str) -> dict | None:
        if not self.live:
            return None
        resp = httpx.get(
            f"{self.url}/rest/v1/{self.table}",
            headers=self._headers(),
            params={"idempotency_key": f"eq.{key}",
                    "select": "idempotency_key,payload,fingerprint"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0]["payload"] if rows else None


def _encode(payload: dict) -> str:
    import json
    return json.dumps(payload)


def _decode(raw: Any) -> dict | None:
    import json
    try:
        return json.loads(raw) if isinstance(raw, str) else None
    except (json.JSONDecodeError, TypeError):
        return None
