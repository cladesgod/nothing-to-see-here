"""Prometheus metrics for the LM-AIG API.

Tracks pipeline runs, queue depth, and rate limiting.
Metrics are exposed via /api/v1/metrics endpoint in Prometheus format.

Uses prometheus_client when available, falls back to no-op counters
if the package is not installed (metrics are optional).
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

try:
    from prometheus_client import Counter, Gauge, generate_latest

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    RUNS_SUBMITTED = Counter(
        "aig_runs_submitted_total",
        "Total pipeline runs submitted",
        ["user_id", "preset", "mode"],
    )
    RUNS_COMPLETED = Counter(
        "aig_runs_completed_total",
        "Total pipeline runs completed",
        ["status"],
    )
    QUEUE_DEPTH = Gauge(
        "aig_queue_depth",
        "Current number of queued runs",
    )
    ACTIVE_WORKERS = Gauge(
        "aig_active_workers",
        "Currently active worker tasks",
    )
    RATE_LIMIT_HITS = Counter(
        "aig_rate_limit_hits_total",
        "Rate limit rejections",
        ["user_id", "limit_type"],
    )


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_run_submitted(user_id: str, preset: str, mode: str) -> None:
    if _PROMETHEUS_AVAILABLE:
        RUNS_SUBMITTED.labels(user_id=user_id, preset=preset, mode=mode).inc()


def record_run_completed(status: str) -> None:
    if _PROMETHEUS_AVAILABLE:
        RUNS_COMPLETED.labels(status=status).inc()


def set_queue_depth(depth: int) -> None:
    if _PROMETHEUS_AVAILABLE:
        QUEUE_DEPTH.set(depth)


def set_active_workers(count: int) -> None:
    if _PROMETHEUS_AVAILABLE:
        ACTIVE_WORKERS.set(count)


def record_rate_limit_hit(user_id: str, limit_type: str) -> None:
    if _PROMETHEUS_AVAILABLE:
        RATE_LIMIT_HITS.labels(user_id=user_id, limit_type=limit_type).inc()


def get_metrics_text() -> str:
    """Generate Prometheus metrics text output."""
    if _PROMETHEUS_AVAILABLE:
        return generate_latest().decode("utf-8")
    return "# prometheus_client not installed\n"
