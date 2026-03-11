"""
Implement a simple token bucket rate limiter to control the rate of API calls.
This is used in various ingest scripts to respect API rate limits and avoid throttling.
""" 

import threading
import time
import logging

logger = logging.getLogger(__name__)

# ----------------------------
# Rate limiter (token bucket)
# ----------------------------
class RateLimiter:
    def __init__(self, rate: float, capacity: int, logger: logging=None) -> None:
        self.rate = float(rate)
        self.capacity = int(capacity)
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()
        self.lock = threading.Lock()
        self.logger = logger or logging.getLogger(__name__)

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.updated_at
                self.updated_at = now

                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return

                missing = 1.0 - self.tokens
                sleep_for = missing / self.rate if self.rate > 0 else 0.1
            self.logger.debug(f"Rate limit exceeded, sleeping for {sleep_for:.2f}s")
            time.sleep(max(0.01, sleep_for))