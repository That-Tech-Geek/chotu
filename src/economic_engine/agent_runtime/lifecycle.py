"""Action lifecycle states — never a binary executed/not."""
from __future__ import annotations

import enum


class ActionState(enum.Enum):
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    QUEUED = "QUEUED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    RETRYABLE = "RETRYABLE"
    BLOCKED = "BLOCKED"
    EXPIRED = "EXPIRED"
