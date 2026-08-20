"""Negotiation state/cache. In-memory by default; swappable for Upstash
Redis-over-REST without a Redis client dep (Vercel bundle diet)."""
from __future__ import annotations

import threading
import time


class StateStore:
    def get(self, key):
        raise NotImplementedError

    def put(self, key, value, ttl: int | None = None):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError


class InMemoryState(StateStore):
    def __init__(self):
        self._data = {}
        self._expiry = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            exp = self._expiry.get(key)
            if exp is not None and exp < time.time():
                self._data.pop(key, None)
                self._expiry.pop(key, None)
                return None
            return self._data.get(key)

    def put(self, key, value, ttl: int | None = None):
        with self._lock:
            self._data[key] = value
            if ttl is not None:
                self._expiry[key] = time.time() + ttl

    def delete(self, key):
        with self._lock:
            self._data.pop(key, None)
            self._expiry.pop(key, None)
