"""IdempotencyStore: real Supabase REST when configured, file fallback otherwise."""
from __future__ import annotations

import httpx
import pytest

from economic_engine.agent_runtime.persistence import IdempotencyStore


def test_file_fallback_records_and_sees(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setenv("IDEMPOTENCY_CACHE_PATH", str(tmp_path / "c.jsonl"))
    store = IdempotencyStore()
    assert store.live is False
    key = store.cache_key("n1", "a1", 0)
    assert store.seen(key) is False
    store.record(key, {"x": 1})
    assert store.seen(key) is True


def test_supabase_live_path(tmp_path, monkeypatch):
    monkeypatch.setenv("IDEMPOTENCY_CACHE_PATH", str(tmp_path / "c.jsonl"))
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.method == "GET":
            return httpx.Response(200, json=[{"idempotency_key": "k"}])
        return httpx.Response(201, json={})

    orig_get, orig_post = httpx.get, httpx.post
    client = httpx.Client(transport=httpx.MockTransport(handler))
    httpx.get, httpx.post = client.get, client.post
    try:
        store = IdempotencyStore(url="https://proj.supabase.co", key="svc")
        assert store.live is True
        assert store.seen("k") is True
        store.record("k2", {"x": 2})
    finally:
        httpx.get, httpx.post = orig_get, orig_post
    assert any(m == "GET" and "action_records" in u for m, u in calls)
    assert any(m == "POST" and "action_records" in u for m, u in calls)


def test_supabase_failure_falls_back_to_file(tmp_path, monkeypatch):
    monkeypatch.setenv("IDEMPOTENCY_CACHE_PATH", str(tmp_path / "c.jsonl"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    orig_get, orig_post = httpx.get, httpx.post
    client = httpx.Client(transport=httpx.MockTransport(handler))
    httpx.get, httpx.post = client.get, client.post
    try:
        store = IdempotencyStore(url="https://proj.supabase.co", key="svc")
        key = store.cache_key("n", "a", 0)
        assert store.seen(key) is False  # HTTP fails → file cache (empty)
        store.record(key, {"x": 1})      # POST fails but file cache holds it
        assert store.seen(key) is True   # file cache catches the record
    finally:
        httpx.get, httpx.post = orig_get, orig_post
