"""Async worker pool for pipeline execution.

Tier 1: In-process async workers with bounded concurrency.
Uses asyncio.Semaphore for global worker limits and per-user limits.

For Tier 2 (distributed), swap to Celery + Redis with the same
WorkerPool interface but backed by task broker.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

import structlog

from src.config import get_agent_settings
from src.graphs.main_workflow import build_main_workflow
from src.persistence.db import get_connection
from src.persistence.repository import create_run, finish_run
from src.schemas.constructs import (
    Construct,
    build_dimension_info,
    compute_fingerprint,
)
from src.schemas.phases import Phase

logger = structlog.get_logger(__name__)


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FEEDBACK = "waiting_feedback"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunConfig:
    """Configuration for a single pipeline run."""

    construct: Construct
    lewmod: bool = False
    max_revisions: int | None = None
    user_id: str = "anonymous"


@dataclass
class RunInfo:
    """Tracked metadata for a pipeline run."""

    run_id: str
    user_id: str
    status: RunStatus
    construct_name: str
    mode: str
    phase: str | None = None
    revision_count: int = 0
    max_revisions: int = 5
    items_text: str | None = None
    review_text: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    error: str | None = None


class WorkerPool:
    """Async worker pool with bounded concurrency.

    Manages pipeline runs as asyncio tasks with:
    - Global max worker limit (prevents resource exhaustion)
    - Per-run tracking (status, results, errors)
    - Graceful cancellation support
    - Automatic cleanup of finished tasks

    Args:
        max_workers: Maximum concurrent pipeline executions.
        db_url: Database URL (SQLite or PostgreSQL).
    """

    def __init__(self, max_workers: int = 10, db_url: str | None = None) -> None:
        self._semaphore = asyncio.Semaphore(max_workers)
        self._max_workers = max_workers
        self._tasks: dict[str, asyncio.Task] = {}
        self._runs: dict[str, RunInfo] = {}
        self._db_url = db_url

    async def submit(self, config: RunConfig) -> str:
        """Submit a new pipeline run.

        Returns the run_id immediately. The pipeline executes
        in the background as an asyncio task.
        """
        run_id = str(uuid.uuid4())
        agent_settings = get_agent_settings()
        max_revisions = config.max_revisions or agent_settings.workflow.max_revisions
        if config.lewmod:
            max_revisions = 999

        run_info = RunInfo(
            run_id=run_id,
            user_id=config.user_id,
            status=RunStatus.QUEUED,
            construct_name=config.construct.name,
            mode="lewmod" if config.lewmod else "human",
            max_revisions=max_revisions,
        )
        self._runs[run_id] = run_info

        task = asyncio.create_task(self._execute(run_id, config, run_info))
        self._tasks[run_id] = task
        task.add_done_callback(lambda t: self._on_task_done(run_id, t))

        logger.info(
            "run_submitted",
            run_id=run_id,
            user_id=config.user_id,
            construct=config.construct.name,
            lewmod=config.lewmod,
        )
        return run_id

    async def _execute(self, run_id: str, config: RunConfig, run_info: RunInfo) -> None:
        """Execute a pipeline run within the semaphore-bounded pool."""
        async with self._semaphore:
            run_info.status = RunStatus.RUNNING
            logger.info("run_started", run_id=run_id)

            agent_settings = get_agent_settings()
            construct = config.construct
            dimension_info = build_dimension_info(construct)
            fingerprint = compute_fingerprint(construct)

            # Persistence setup
            conn = None
            try:
                conn = get_connection(self._db_url)
                create_run(
                    conn,
                    run_id=run_id,
                    construct_name=construct.name,
                    construct_definition=construct.definition,
                    construct_fingerprint=fingerprint,
                    mode=run_info.mode,
                    model=agent_settings.defaults.model,
                    max_revisions=run_info.max_revisions,
                )
                db_row = conn.execute("PRAGMA database_list").fetchone()
                db_path = str(db_row["file"]) if db_row and db_row["file"] else self._db_url

                # Build and run graph
                graph = build_main_workflow(lewmod=config.lewmod)
                graph_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

                initial_state = {
                    "construct_name": construct.name,
                    "construct_definition": construct.definition,
                    "dimension_info": dimension_info,
                    "construct_fingerprint": fingerprint,
                    "current_phase": Phase.WEB_RESEARCH,
                    "revision_count": 0,
                    "max_revisions": run_info.max_revisions,
                    "run_id": run_id,
                    "db_path": db_path,
                    "previously_approved_items": [],
                    "human_item_decisions": {},
                    "human_global_note": "",
                    "messages": [],
                }

                async for _event in graph.astream(initial_state, graph_config, stream_mode="updates"):
                    # Update phase from events
                    if isinstance(_event, dict):
                        for _node_name, node_output in _event.items():
                            if isinstance(node_output, dict):
                                if "current_phase" in node_output:
                                    run_info.phase = node_output["current_phase"]
                                if "items_text" in node_output:
                                    run_info.items_text = node_output["items_text"]
                                if "review_text" in node_output:
                                    run_info.review_text = node_output["review_text"]
                                if "revision_count" in node_output:
                                    run_info.revision_count = node_output["revision_count"]

                # Finalize
                final_state = graph.get_state(graph_config).values
                run_info.items_text = final_state.get("items_text")
                run_info.review_text = final_state.get("review_text")
                run_info.revision_count = final_state.get("revision_count", 0)
                run_info.status = RunStatus.DONE
                run_info.finished_at = datetime.now(timezone.utc)

                finish_run(conn, run_id, status="done", total_revisions=run_info.revision_count)
                logger.info("run_completed", run_id=run_id, revisions=run_info.revision_count)

            except asyncio.CancelledError:
                run_info.status = RunStatus.CANCELLED
                run_info.finished_at = datetime.now(timezone.utc)
                if conn:
                    finish_run(conn, run_id, status="cancelled")
                logger.info("run_cancelled", run_id=run_id)
                raise

            except Exception as exc:
                run_info.status = RunStatus.FAILED
                run_info.error = str(exc)
                run_info.finished_at = datetime.now(timezone.utc)
                if conn:
                    finish_run(conn, run_id, status="failed")
                logger.error("run_failed", run_id=run_id, error=str(exc), exc_info=True)

            finally:
                if conn:
                    conn.close()

    def _on_task_done(self, run_id: str, task: asyncio.Task) -> None:
        """Callback when a task completes (success, failure, or cancellation)."""
        self._tasks.pop(run_id, None)

    def get_run(self, run_id: str) -> RunInfo | None:
        """Get run info by ID."""
        return self._runs.get(run_id)

    def list_runs(
        self,
        user_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[RunInfo], int]:
        """List runs, optionally filtered by user. Returns (runs, total)."""
        runs = list(self._runs.values())
        if user_id:
            runs = [r for r in runs if r.user_id == user_id]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        total = len(runs)
        start = (page - 1) * page_size
        return runs[start : start + page_size], total

    async def cancel(self, run_id: str) -> bool:
        """Cancel a running pipeline. Returns True if cancelled."""
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    @property
    def pending_count(self) -> int:
        """Number of tasks waiting for a worker slot."""
        running = sum(1 for t in self._tasks.values() if not t.done())
        return max(0, running - self._max_workers)

    @property
    def active_count(self) -> int:
        """Number of currently executing tasks."""
        return sum(1 for t in self._tasks.values() if not t.done())
