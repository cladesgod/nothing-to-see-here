"""FastAPI dependency injection for the LM-AIG API.

Provides reusable dependencies for authentication, rate limiting,
and the worker pool. All dependencies are initialized once at startup
and injected into route handlers via FastAPI's Depends().
"""

from __future__ import annotations

from fastapi import Header, HTTPException, Request

from src.api.auth import APIKeyAuth, APIUser
from src.api.metrics import record_rate_limit_hit
from src.api.queue import WorkerPool
from src.api.rate_limiter import RateLimiter, UserConcurrencyLimiter


# ---------------------------------------------------------------------------
# Singleton instances (initialized in app lifespan)
# ---------------------------------------------------------------------------

_auth: APIKeyAuth | None = None
_rate_limiter: RateLimiter | None = None
_concurrency_limiter: UserConcurrencyLimiter | None = None
_worker_pool: WorkerPool | None = None


def init_dependencies(
    auth: APIKeyAuth,
    rate_limiter: RateLimiter,
    concurrency_limiter: UserConcurrencyLimiter,
    worker_pool: WorkerPool,
) -> None:
    """Initialize shared dependency instances. Called once at app startup."""
    global _auth, _rate_limiter, _concurrency_limiter, _worker_pool
    _auth = auth
    _rate_limiter = rate_limiter
    _concurrency_limiter = concurrency_limiter
    _worker_pool = worker_pool


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def get_current_user(x_api_key: str = Header(..., alias="X-API-Key")) -> APIUser:
    """Authenticate the request via API key header.

    Raises 401 if the key is missing or invalid.
    """
    if _auth is None:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    user = _auth.verify(x_api_key)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return user


async def check_rate_limit(request: Request, user: APIUser) -> None:
    """Check per-user rate limits. Raises 429 if exceeded.

    Should be called after authentication.
    """
    if _rate_limiter is None:
        return

    allowed, retry_after = _rate_limiter.check(user.user_id)
    if not allowed:
        record_rate_limit_hit(user.user_id, "request_rate")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


async def check_concurrency(user: APIUser) -> None:
    """Check per-user concurrency limit. Raises 429 if at capacity.

    Should be called before submitting a new run.
    """
    if _concurrency_limiter is None:
        return

    acquired = await _concurrency_limiter.acquire(user.user_id)
    if not acquired:
        record_rate_limit_hit(user.user_id, "concurrency")
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent runs. Wait for a run to complete.",
        )


def get_worker_pool() -> WorkerPool:
    """Get the shared worker pool instance."""
    if _worker_pool is None:
        raise HTTPException(status_code=500, detail="Worker pool not initialized")
    return _worker_pool


def get_concurrency_limiter() -> UserConcurrencyLimiter:
    """Get the shared concurrency limiter instance."""
    if _concurrency_limiter is None:
        raise HTTPException(status_code=500, detail="Concurrency limiter not initialized")
    return _concurrency_limiter
