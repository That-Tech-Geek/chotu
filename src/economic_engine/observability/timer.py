"""Lightweight timing / benchmark instrumentation."""
from __future__ import annotations

import contextlib
import time


class Timer:
    @contextlib.contextmanager
    def time(self, name: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.last = {**(getattr(self, "last", {})), name: elapsed}

    def __init__(self):
        self.last: dict[str, float] = {}
