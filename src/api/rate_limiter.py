"""Rate limiting and concurrency control for the LM-AIG API.

Provides two complementary mechanisms:
1. RateLimiter — token-bucket per-user request rate limiting
2. UserConcurrencyLimiter — per-user concurrent run limits

Both are in-memory (Tier 1). For Tier 2 (distributed), swap to Redis-backed
implementations using the same interface.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter
# ---------------------------------------------------------------------------


@dataclass
class _TokenBucket:
    """Token bucket for rate limiting a single user."""

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = self.capacity

    def consume(self) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def retry_after(self) -> float:
        """Seconds until the next token is available."""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.refill_rate


class RateLimiter:
    """Per-user request rate limiting using token bucket algorithm.

    Each user gets an independent bucket with configurable capacity
    and refill rate. Buckets are created lazily on first request.

    Args:
        requests_per_minute: Maximum burst capacity per user.
        requests_per_day: Daily quota (enforced via a separate slow bucket).
    """

    def __init__(
        self,
        requests_per_minute: int = 10,
        requests_per_day: int = 100,
    ) -> None:
        self._rpm = requests_per_minute
        self._daily = requests_per_day
        # Fast bucket: per-minute burst control
        self._minute_buckets: dict[str, _TokenBucket] = {}
        # Slow bucket: daily quota
        self._daily_buckets: dict[str, _TokenBucket] = {}

    def check(self, user_id: str) -> tuple[bool, float]:
        """Check if a request is allowed for this user.

        Returns:
            (allowed, retry_after_seconds)
        """
        # Minute-level rate limit
        if user_id not in self._minute_buckets:
            self._minute_buckets[user_id] = _TokenBucket(
                capacity=float(self._rpm),
                refill_rate=self._rpm / 60.0,
            )
        minute_bucket = self._minute_buckets[user_id]

        # Daily quota
        if user_id not in self._daily_buckets:
            self._daily_buckets[user_id] = _TokenBucket(
                capacity=float(self._daily),
                refill_rate=self._daily / 86400.0,
            )
        daily_bucket = self._daily_buckets[user_id]

        if not daily_bucket.consume():
            logger.warning("rate_limit_daily_exceeded", user_id=user_id)
            return False, daily_bucket.retry_after

        if not minute_bucket.consume():
            # Refund the daily token since we're rejecting
            daily_bucket.tokens = min(daily_bucket.capacity, daily_bucket.tokens + 1.0)
            logger.warning("rate_limit_minute_exceeded", user_id=user_id)
            return False, minute_bucket.retry_after

        return True, 0.0


# ---------------------------------------------------------------------------
# Per-User Concurrency Limiter
# ---------------------------------------------------------------------------


class UserConcurrencyLimiter:
    """Track and limit concurrent pipeline runs per user.

    In-memory implementation (Tier 1). For distributed deployments,
    replace with Redis INCR/DECR-based implementation.
    """

    def __init__(self, max_concurrent: int = 3) -> None:
        self._max = max_concurrent
        self._counts: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, user_id: str) -> bool:
        """Try to acquire a concurrency slot for the user.

        Returns True if the user has capacity, False if at limit.
        """
        async with self._lock:
            current = self._counts.get(user_id, 0)
            if current >= self._max:
                logger.warning(
                    "concurrency_limit_reached",
                    user_id=user_id,
                    current=current,
                    max=self._max,
                )
                return False
            self._counts[user_id] = current + 1
            return True

    async def release(self, user_id: str) -> None:
        """Release a concurrency slot for the user."""
        async with self._lock:
            current = self._counts.get(user_id, 0)
            if current > 0:
                self._counts[user_id] = current - 1

    def active_count(self, user_id: str) -> int:
        """Get the number of active runs for a user."""
        return self._counts.get(user_id, 0)

    @property
    def total_active(self) -> int:
        """Total active runs across all users."""
        return sum(self._counts.values())
