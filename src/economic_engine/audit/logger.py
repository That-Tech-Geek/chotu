"""Audit trail: append-only event records."""
from __future__ import annotations

import time


class AuditLog:
    def __init__(self):
        self.events: list[dict] = []

    def record(self, event_type: str, payload: dict) -> None:
        self.events.append({"type": event_type, "at": time.time(), **payload})
