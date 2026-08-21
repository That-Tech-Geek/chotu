"""Atomic claim primitives: Upstash SET NX and Supabase ignore-duplicates,
verified through the real HTTP code paths (network faked by MockTransport)."""
from __future__ import annotations

import json

import httpx

from economic_engine.agent_runtime.persistence import SupabaseActionLog, UpstashLock


def _swap(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    orig_get, orig_post = httpx.get, httpx.post
    httpx.get, httpx.post = client.get, client.post
    return orig_get, orig_post, client


def _restore(orig_get, orig_post):
    httpx.get, httpx.post = orig_get, orig_post


def test_upstash_claim_single_winner():
    calls = []
    claims = {"winner": None}

    def handler(request: httpx.Request) -> httpx.Response:
        cmd = json.loads(request.content)
        calls.append(cmd)
        if cmd[0] == "SET":
            if claims["winner"] is None:
                claims["winner"] = cmd[2]
                return httpx.Response(200, json={"result": "OK"})
            return httpx.Response(200, json={"result": None})
        if cmd[0] == "GET":
            return httpx.Response(200, json={"result": claims["winner"]})
        return httpx.Response(200, json={"result": 1})

    orig_get, orig_post, _ = _swap(handler)
    try:
        lock = UpstashLock(url="https://upstash.example.com", token="tok")
        r1 = lock.claim("k1", {"fingerprint": "fp1"})
        r2 = lock.claim("k1", {"fingerprint": "fp2"})
        assert r1.claimed is True
        assert r2.claimed is False and r2.reason == "already_claimed"
        assert r2.owner_payload == {"fingerprint": "fp1"}
        set_cmd = next(c for c in calls if c[0] == "SET")
        assert "NX" in set_cmd and "EX" in set_cmd  # the atomic primitive
    finally:
        _restore(orig_get, orig_post)


def test_upstash_release_only_owner():
    deleted = []
    state = {"key": json.dumps({"fingerprint": "fp1"})}

    def handler(request: httpx.Request) -> httpx.Response:
        cmd = json.loads(request.content)
        if cmd[0] == "GET":
            return httpx.Response(200, json={"result": state["key"]})
        if cmd[0] == "DEL":
            deleted.append(cmd[1])
            return httpx.Response(200, json={"result": 1})
        return httpx.Response(200, json={"result": "OK"})

    orig_get, orig_post, _ = _swap(handler)
    try:
        lock = UpstashLock(url="https://upstash.example.com", token="tok")
        lock.release("key", {"fingerprint": "fp-other"})
        assert deleted == []  # non-owner cannot release
        lock.release("key", {"fingerprint": "fp1"})
        assert deleted == ["key"]
    finally:
        _restore(orig_get, orig_post)


def test_upstash_not_configured_refuses_claim():
    lock = UpstashLock(url="", token="")
    r = lock.claim("k", {"fingerprint": "fp"})
    assert r.claimed is False


def test_supabase_claim_uses_ignore_duplicates():
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            201,
            json=[{"idempotency_key": "k", "fingerprint": "fp1"}],
        )

    orig_get, orig_post, _ = _swap(handler)
    try:
        ledger = SupabaseActionLog(url="https://proj.supabase.co", key="svc")
        r = ledger.claim("k", {"fingerprint": "fp1"})
        assert r.claimed is True
        req = captured[0]
        assert "resolution=ignore-duplicates" in req.headers["Prefer"]
    finally:
        _restore(orig_get, orig_post)


def test_supabase_claim_loser_when_fingerprint_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json=[{"idempotency_key": "k", "fingerprint": "fp-other"}],
        )

    orig_get, orig_post, _ = _swap(handler)
    try:
        ledger = SupabaseActionLog(url="https://proj.supabase.co", key="svc")
        r = ledger.claim("k", {"fingerprint": "fp1"})
        assert r.claimed is False and r.reason == "already_claimed"
    finally:
        _restore(orig_get, orig_post)
