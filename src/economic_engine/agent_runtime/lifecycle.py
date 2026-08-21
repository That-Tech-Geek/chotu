"""Action lifecycle states — never a binary executed/not."""
from __future__ import annotations

import enum


class ActionState(enum.Enum):
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    QUEUED = "QUEUED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"          # provider definitively refused
    UNKNOWN = "UNKNOWN"        # network timeout / no response — effect uncertain
    RETRYABLE = "RETRYABLE"
    BLOCKED = "BLOCKED"
    EXPIRED = "EXPIRED"
