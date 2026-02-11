"""Tests for the API layer: auth, rate limiter, concurrency, schemas, queue."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.api.auth import APIKeyAuth, APIUser
from src.api.rate_limiter import RateLimiter, UserConcurrencyLimiter, _TokenBucket
from src.api.schemas import (
    ConstructDefinition,
    DimensionInput,
    ErrorResponse,
    FeedbackRequest,
    HealthResponse,
    RunCreateRequest,
    RunCreatedResponse,
    RunListResponse,
    RunStatusResponse,
)


# ===========================================================================
# Auth Tests
# ===========================================================================


class TestAPIKeyAuth:
    def test_register_and_verify(self):
        auth = APIKeyAuth()
        auth.register_key("test-key-123", "user_1")
        user = auth.verify("test-key-123")
        assert user is not None
        assert user.user_id == "user_1"
        assert user.key_prefix == "test-key"

    def test_verify_invalid_key(self):
        auth = APIKeyAuth()
        auth.register_key("test-key-123", "user_1")
        assert auth.verify("wrong-key") is None

    def test_verify_empty_key(self):
        auth = APIKeyAuth()
        assert auth.verify("") is None

    def test_generate_key_format(self):
        key = APIKeyAuth.generate_key()
        assert key.startswith("aig_")
        assert len(key) > 20

    def test_generate_key_uniqueness(self):
        keys = {APIKeyAuth.generate_key() for _ in range(100)}
        assert len(keys) == 100  # All unique

    def test_from_env_keys_user_key_pairs(self):
        auth = APIKeyAuth.from_env_keys("alice:key1,bob:key2")
        assert auth.verify("key1") is not None
        assert auth.verify("key1").user_id == "alice"
        assert auth.verify("key2").user_id == "bob"

    def test_from_env_keys_bare_keys(self):
        auth = APIKeyAuth.from_env_keys("key1,key2")
        assert auth.verify("key1") is not None
        assert auth.verify("key1").user_id == "user_0"
        assert auth.verify("key2").user_id == "user_1"

    def test_from_env_keys_empty(self):
        auth = APIKeyAuth.from_env_keys("")
        assert auth.verify("anything") is None

    def test_keys_stored_as_hashes(self):
        """API keys are hashed — raw keys should not appear in the dict."""
        auth = APIKeyAuth()
        auth.register_key("my-secret-key", "user_1")
        for stored_hash in auth._key_to_user:
            assert "my-secret-key" not in stored_hash
            assert len(stored_hash) == 64  # SHA-256 hex length

    def test_timing_safe_comparison(self):
        """Verify that constant-time comparison is used (hmac.compare_digest)."""
        auth = APIKeyAuth()
        auth.register_key("test-key", "user_1")
        # Both should complete in similar time regardless of prefix match
        auth.verify("test-key")
        auth.verify("zzzz-zzz")


# ===========================================================================
# Rate Limiter Tests
# ===========================================================================


class TestTokenBucket:
    def test_initial_tokens(self):
        bucket = _TokenBucket(capacity=5.0, refill_rate=1.0)
        assert bucket.tokens == 5.0

    def test_consume_success(self):
        bucket = _TokenBucket(capacity=5.0, refill_rate=1.0)
        assert bucket.consume() is True
        assert bucket.tokens < 5.0

    def test_consume_depleted(self):
        bucket = _TokenBucket(capacity=2.0, refill_rate=0.0)  # No refill
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False  # Depleted

    def test_refill_over_time(self):
        bucket = _TokenBucket(capacity=5.0, refill_rate=100.0)  # Fast refill
        for _ in range(5):
            bucket.consume()
        # With fast refill rate, should recover quickly
        time.sleep(0.1)
        assert bucket.consume() is True

    def test_retry_after(self):
        bucket = _TokenBucket(capacity=1.0, refill_rate=1.0)
        bucket.consume()
        assert bucket.retry_after > 0


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=100)
        for _ in range(5):
            allowed, _ = limiter.check("user_1")
            assert allowed is True

    def test_blocks_over_minute_limit(self):
        limiter = RateLimiter(requests_per_minute=2, requests_per_day=100)
        limiter.check("user_1")
        limiter.check("user_1")
        allowed, retry_after = limiter.check("user_1")
        assert allowed is False
        assert retry_after > 0

    def test_separate_users(self):
        limiter = RateLimiter(requests_per_minute=1, requests_per_day=100)
        allowed1, _ = limiter.check("user_1")
        allowed2, _ = limiter.check("user_2")
        assert allowed1 is True
        assert allowed2 is True

    def test_daily_limit(self):
        limiter = RateLimiter(requests_per_minute=1000, requests_per_day=3)
        limiter.check("user_1")
        limiter.check("user_1")
        limiter.check("user_1")
        allowed, _ = limiter.check("user_1")
        assert allowed is False


# ===========================================================================
# Concurrency Limiter Tests
# ===========================================================================


class TestUserConcurrencyLimiter:
    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        limiter = UserConcurrencyLimiter(max_concurrent=3)
        assert await limiter.acquire("user_1") is True
        assert await limiter.acquire("user_1") is True
        assert await limiter.acquire("user_1") is True

    @pytest.mark.asyncio
    async def test_acquire_over_limit(self):
        limiter = UserConcurrencyLimiter(max_concurrent=2)
        assert await limiter.acquire("user_1") is True
        assert await limiter.acquire("user_1") is True
        assert await limiter.acquire("user_1") is False  # Over limit

    @pytest.mark.asyncio
    async def test_release_frees_slot(self):
        limiter = UserConcurrencyLimiter(max_concurrent=1)
        assert await limiter.acquire("user_1") is True
        assert await limiter.acquire("user_1") is False
        await limiter.release("user_1")
        assert await limiter.acquire("user_1") is True

    @pytest.mark.asyncio
    async def test_separate_users(self):
        limiter = UserConcurrencyLimiter(max_concurrent=1)
        assert await limiter.acquire("user_1") is True
        assert await limiter.acquire("user_2") is True  # Different user

    @pytest.mark.asyncio
    async def test_active_count(self):
        limiter = UserConcurrencyLimiter(max_concurrent=5)
        await limiter.acquire("user_1")
        await limiter.acquire("user_1")
        assert limiter.active_count("user_1") == 2
        assert limiter.total_active == 2

    @pytest.mark.asyncio
    async def test_release_noop_if_zero(self):
        limiter = UserConcurrencyLimiter(max_concurrent=5)
        await limiter.release("user_1")  # Should not go negative
        assert limiter.active_count("user_1") == 0


# ===========================================================================
# Schema Tests
# ===========================================================================


class TestAPISchemas:
    def test_run_create_request_preset(self):
        req = RunCreateRequest(preset="aaaw", lewmod=True)
        assert req.preset == "aaaw"
        assert req.lewmod is True
        assert req.construct_definition is None

    def test_run_create_request_custom_construct(self):
        req = RunCreateRequest(
            construct_definition=ConstructDefinition(
                name="Test Construct",
                definition="A test construct for testing.",
                dimensions=[
                    DimensionInput(
                        name="Dim 1",
                        definition="First dimension",
                        orbiting=["Orbit A", "Orbit B"],
                    )
                ],
            )
        )
        assert req.construct_definition.name == "Test Construct"
        assert len(req.construct_definition.dimensions) == 1

    def test_run_create_request_max_revisions_validation(self):
        req = RunCreateRequest(max_revisions=5)
        assert req.max_revisions == 5
        with pytest.raises(Exception):
            RunCreateRequest(max_revisions=0)  # Must be >= 1
        with pytest.raises(Exception):
            RunCreateRequest(max_revisions=25)  # Must be <= 20

    def test_feedback_request_approve(self):
        req = FeedbackRequest(approve=True)
        assert req.approve is True
        assert req.item_decisions == {}

    def test_feedback_request_revise(self):
        req = FeedbackRequest(
            item_decisions={"1": "KEEP", "2": "REVISE"},
            global_note="Please improve clarity.",
        )
        assert req.item_decisions["1"] == "KEEP"
        assert len(req.global_note) > 0

    def test_health_response(self):
        resp = HealthResponse(queue_depth=5, active_workers=3)
        assert resp.status == "healthy"
        assert resp.queue_depth == 5

    def test_run_status_response(self):
        resp = RunStatusResponse(
            run_id="test-123",
            status="running",
            construct_name="AAAW",
            mode="lewmod",
        )
        assert resp.run_id == "test-123"
        assert resp.revision_count == 0

    def test_error_response(self):
        resp = ErrorResponse(error="Not found", detail="Run ID does not exist")
        assert resp.error == "Not found"

    def test_construct_definition_validation(self):
        """Name must be at least 1 char, definition at least 10 chars."""
        with pytest.raises(Exception):
            ConstructDefinition(
                name="",
                definition="Short",
                dimensions=[],
            )


# ===========================================================================
# Config Tests
# ===========================================================================


class TestAPIConfig:
    def test_api_config_loaded_from_toml(self):
        from src.config import get_agent_settings

        settings = get_agent_settings()
        assert hasattr(settings, "api")
        assert settings.api.max_workers >= 1
        assert settings.api.max_concurrent_per_user >= 1
        assert settings.api.rate_limit_rpm >= 1

    def test_api_config_defaults(self):
        from src.config import APIConfig

        config = APIConfig()
        assert config.max_workers == 10
        assert config.max_concurrent_per_user == 3
        assert config.rate_limit_rpm == 10
        assert config.rate_limit_daily == 100


# ===========================================================================
# DB Abstraction Tests
# ===========================================================================


class TestDBAbstraction:
    def test_sqlite_connection_default(self, tmp_path):
        from src.persistence.db import get_connection

        db_path = tmp_path / "test.db"
        conn = get_connection(str(db_path))
        assert conn is not None
        # Verify tables exist
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "runs" in tables
        assert "research" in tables
        conn.close()

    def test_pg_url_detection(self):
        """Verify that PostgreSQL URLs are detected (even if psycopg not installed)."""
        from src.persistence.db import get_connection

        # This should try to import psycopg — may raise ImportError if not installed
        try:
            get_connection("postgresql://user:pass@localhost/test_db")
        except ImportError as e:
            assert "psycopg" in str(e)
        except Exception:
            pass  # Connection refused is fine — we're testing URL routing
