"""Durable idempotency: Supabase REST first (survives process death), with a
file-cache fallback for local/test. Never RAM-only."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx


class IdempotencyStore:
    def __init__(self, url: str | None = None, key: str | None = None,
                 table: str = "action_records", timeout: float = 5.0):
        self.url = (url if url is not None else os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = key if key is not None else os.environ.get("SUPABASE_KEY", "")
        self.table = table
        self.timeout = timeout
        self.cache_path = Path(
            os.environ.get("IDEMPOTENCY_CACHE_PATH", "/tmp/idempotency_cache.jsonl")
        )

    @property
    def live(self) -> bool:
        return bool(self.url and self.key)

    def cache_key(self, negotiation_id: str, action_id: str, sequence: int) -> str:
        return f"{negotiation_id}:{action_id}:{sequence}"

    def _headers(self) -> dict:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    def seen(self, key: str, ttl_seconds: int = 86400) -> bool:
        if self.live:
            try:
                resp = httpx.get(
                    f"{self.url}/rest/v1/{self.table}",
                    headers=self._headers(),
                    params={"idempotency_key": f"eq.{key}", "select": "idempotency_key,created_at"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                rows = resp.json()
                if rows:
                    return True
            except httpx.HTTPError:
                # Fall through to the file cache — never silently pass.
                pass
        if not self.cache_path.parent.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            return False
        cutoff = time.time() - ttl_seconds
        with self.cache_path.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("key") == key and rec.get("ts", 0) > cutoff:
                    return True
        return False

    def record(self, key: str, record: dict) -> None:
        record["key"] = key
        record["ts"] = time.time()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if self.live:
            try:
                httpx.post(
                    f"{self.url}/rest/v1/{self.table}",
                    headers={**self._headers(), "Prefer": "return=minimal"},
                    json={
                        "idempotency_key": key,
                        "payload": record,
                    },
                    timeout=self.timeout,
                )
            except httpx.HTTPError:
                # The file cache already holds it; next seen() still catches.
                pass
