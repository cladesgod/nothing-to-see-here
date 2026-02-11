"""FastAPI application for the LM-AIG Multi-Agent System.

Provides REST API endpoints for submitting pipeline runs, checking status,
submitting human feedback, and monitoring system health.

Usage:
    uvicorn src.api.app:app --reload          # Development
    uvicorn src.api.app:app --host 0.0.0.0    # Production (behind reverse proxy)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.auth import APIKeyAuth, APIUser
from src.api.dependencies import (
    check_concurrency,
    check_rate_limit,
    get_concurrency_limiter,
    get_current_user,
    get_worker_pool,
    init_dependencies,
)
from src.api.metrics import (
    get_metrics_text,
    record_run_completed,
    record_run_submitted,
    set_active_workers,
    set_queue_depth,
)
from src.api.queue import RunConfig, RunStatus, WorkerPool
from src.api.rate_limiter import RateLimiter, UserConcurrencyLimiter
from src.api.schemas import (
    ErrorResponse,
    FeedbackRequest,
    HealthResponse,
    RunCreatedResponse,
    RunListResponse,
    RunStatusResponse,
    RunCreateRequest,
)
from src.config import get_agent_settings
from src.schemas.constructs import Construct, ConstructDimension, get_preset

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# App lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup, clean up on shutdown."""
    agent_settings = get_agent_settings()
    api_config = getattr(agent_settings, "api", None)

    max_workers = int(os.environ.get("AIG_MAX_WORKERS", "10"))
    max_concurrent = int(os.environ.get("AIG_MAX_CONCURRENT_PER_USER", "3"))
    rate_rpm = int(os.environ.get("AIG_RATE_LIMIT_RPM", "10"))
    rate_daily = int(os.environ.get("AIG_RATE_LIMIT_DAILY", "100"))
    db_url = os.environ.get("DATABASE_URL")

    # Initialize auth from env: AIG_API_KEYS="user1:key1,user2:key2"
    api_keys_csv = os.environ.get("AIG_API_KEYS", "")
    auth = APIKeyAuth.from_env_keys(api_keys_csv)

    # If no keys configured, register a dev key for testing
    if not api_keys_csv:
        dev_key = os.environ.get("AIG_DEV_API_KEY", "dev-key-for-testing")
        auth.register_key(dev_key, "dev")
        logger.warning("no_api_keys_configured", dev_key_active=True)

    rate_limiter = RateLimiter(
        requests_per_minute=rate_rpm,
        requests_per_day=rate_daily,
    )
    concurrency_limiter = UserConcurrencyLimiter(max_concurrent=max_concurrent)
    worker_pool = WorkerPool(max_workers=max_workers, db_url=db_url)

    init_dependencies(auth, rate_limiter, concurrency_limiter, worker_pool)
    logger.info(
        "api_started",
        max_workers=max_workers,
        max_concurrent_per_user=max_concurrent,
        rate_limit_rpm=rate_rpm,
    )

    yield

    # Shutdown: cancel remaining tasks
    for run_id in list(worker_pool._tasks.keys()):
        await worker_pool.cancel(run_id)
    logger.info("api_shutdown")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LM-AIG Multi-Agent Item Generation API",
    description=(
        "REST API for submitting and managing multi-agent pipeline runs "
        "for psychological test item generation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — configurable via AIG_CORS_ORIGINS env var
cors_origins = os.environ.get("AIG_CORS_ORIGINS", "").split(",")
cors_origins = [o.strip() for o in cors_origins if o.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/runs",
    response_model=RunCreatedResponse,
    status_code=202,
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def create_run(
    request: RunCreateRequest,
    user: APIUser = Depends(get_current_user),
    pool: WorkerPool = Depends(get_worker_pool),
):
    """Submit a new pipeline run.

    Provide either a `preset` name (e.g., "aaaw") or a custom `construct`
    definition. The run executes asynchronously — poll GET /runs/{id} for status.
    """
    # Rate limit check
    await check_rate_limit(None, user)

    # Concurrency check
    await check_concurrency(user)

    # Resolve construct
    if request.preset and request.construct_definition:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'preset' or 'construct', not both.",
        )

    if request.construct_definition:
        # Custom construct from request body
        dimensions = [
            ConstructDimension(
                name=d.name,
                definition=d.definition,
                orbiting=d.orbiting,
            )
            for d in request.construct_definition.dimensions
        ]
        construct = Construct(
            name=request.construct_definition.name,
            definition=request.construct_definition.definition,
            dimensions=dimensions,
        )
    else:
        # Built-in preset (default: aaaw)
        preset_name = request.preset or "aaaw"
        construct = get_preset(preset_name)
        if construct is None:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown preset: {preset_name}",
            )

    config = RunConfig(
        construct=construct,
        lewmod=request.lewmod,
        max_revisions=request.max_revisions,
        user_id=user.user_id,
    )

    run_id = await pool.submit(config)
    record_run_submitted(user.user_id, request.preset or "custom", "lewmod" if config.lewmod else "human")
    set_queue_depth(pool.pending_count)
    set_active_workers(pool.active_count)

    return RunCreatedResponse(run_id=run_id)


@app.get(
    "/api/v1/runs/{run_id}",
    response_model=RunStatusResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_run(
    run_id: str,
    user: APIUser = Depends(get_current_user),
    pool: WorkerPool = Depends(get_worker_pool),
):
    """Get the status and results of a pipeline run."""
    run_info = pool.get_run(run_id)
    if run_info is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    # Users can only see their own runs
    if run_info.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found.")

    return RunStatusResponse(
        run_id=run_info.run_id,
        status=run_info.status.value,
        phase=run_info.phase,
        construct_name=run_info.construct_name,
        mode=run_info.mode,
        revision_count=run_info.revision_count,
        max_revisions=run_info.max_revisions,
        items_text=run_info.items_text,
        review_text=run_info.review_text,
        created_at=run_info.created_at,
        finished_at=run_info.finished_at,
    )


@app.get(
    "/api/v1/runs",
    response_model=RunListResponse,
    responses={401: {"model": ErrorResponse}},
)
async def list_runs(
    page: int = 1,
    page_size: int = 20,
    user: APIUser = Depends(get_current_user),
    pool: WorkerPool = Depends(get_worker_pool),
):
    """List the current user's pipeline runs (paginated)."""
    runs, total = pool.list_runs(user_id=user.user_id, page=page, page_size=page_size)
    return RunListResponse(
        runs=[
            RunStatusResponse(
                run_id=r.run_id,
                status=r.status.value,
                phase=r.phase,
                construct_name=r.construct_name,
                mode=r.mode,
                revision_count=r.revision_count,
                max_revisions=r.max_revisions,
                items_text=r.items_text,
                review_text=r.review_text,
                created_at=r.created_at,
                finished_at=r.finished_at,
            )
            for r in runs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@app.post(
    "/api/v1/runs/{run_id}/feedback",
    response_model=RunStatusResponse,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def submit_feedback(
    run_id: str,
    feedback: FeedbackRequest,
    user: APIUser = Depends(get_current_user),
    pool: WorkerPool = Depends(get_worker_pool),
):
    """Submit human feedback for a paused run.

    Only works when the run is in 'waiting_feedback' status.
    """
    run_info = pool.get_run(run_id)
    if run_info is None or run_info.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found.")

    if run_info.status != RunStatus.WAITING_FEEDBACK:
        raise HTTPException(
            status_code=409,
            detail=f"Run is not waiting for feedback (status: {run_info.status.value}).",
        )

    # TODO: Resume the graph with feedback via Command(resume=...)
    # This requires the graph checkpointer + thread_id to be accessible
    raise HTTPException(
        status_code=501,
        detail="Feedback submission via API is planned for Phase 2 (PostgreSQL checkpointer).",
    )


@app.delete(
    "/api/v1/runs/{run_id}",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def cancel_run(
    run_id: str,
    user: APIUser = Depends(get_current_user),
    pool: WorkerPool = Depends(get_worker_pool),
):
    """Cancel a running pipeline."""
    run_info = pool.get_run(run_id)
    if run_info is None or run_info.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found.")

    cancelled = await pool.cancel(run_id)
    if not cancelled:
        raise HTTPException(status_code=409, detail="Run is not cancellable.")

    record_run_completed("cancelled")
    # Release concurrency slot
    limiter = get_concurrency_limiter()
    await limiter.release(user.user_id)

    return {"run_id": run_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# Health & Metrics
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    pool = get_worker_pool()
    return HealthResponse(
        status="healthy",
        queue_depth=pool.pending_count,
        active_workers=pool.active_count,
        max_workers=pool._max_workers,
    )


@app.get("/api/v1/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics_text(),
        media_type="text/plain; charset=utf-8",
    )
