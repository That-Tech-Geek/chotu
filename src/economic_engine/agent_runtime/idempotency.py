"""Idempotent action executor — retries are safe; network timeout -> retry
never becomes SEND SEND."""
from __future__ import annotations

import time

from economic_engine.state.canonical import Deal


class ActionExecutor:
    def __init__(self):
        self.seen: dict[str, dict] = {}

    def execute(
        self,
        negotiation_id: str,
        action_id: str,
        sequence: int,
        payload: Deal,
        ttl_seconds: int = 3600,
    ) -> dict:
        key = f"{negotiation_id}:{action_id}:{sequence}"
        if key in self.seen:
            return {
                "executed": False,
                "duplicate": True,
                "idempotency_key": key,
            }
        self.seen[key] = {
            "payload": payload.model_dump(),
            "timestamp": time.time(),
        }
        return {
            "executed": True,
            "duplicate": False,
            "idempotency_key": key,
            "payload": payload.model_dump(),
        }
