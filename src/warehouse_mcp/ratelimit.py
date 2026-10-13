"""Per-principal token bucket, so one runaway agent loop cannot monopolise the warehouse."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self, capacity: int = 60, refill_per_s: float = 1.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if capacity < 1 or refill_per_s <= 0:
            raise ValueError("capacity must be >= 1 and refill_per_s > 0")
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_s)
            if tokens < 1:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1, now)
            return True
