"""Repository functions for pipeline persistence.

Each function takes a sqlite3.Connection and performs a single operation.
Connections are opened/closed by callers (agent nodes or run.py).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


def _now() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def create_run(
    conn: sqlite3.Connection,
    run_id: str,
    construct_name: str,
    construct_definition: str,
    construct_fingerprint: str,
    mode: str,
    model: str,
    max_revisions: int,
) -> str:
    """Create a new pipeline run record. Returns run_id."""
    conn.execute(
        """INSERT INTO runs (id, construct_name, construct_definition,
           construct_fingerprint, mode, model, max_revisions, started_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, construct_name, construct_definition, construct_fingerprint,
         mode, model, max_revisions, _now()),
    )
    conn.commit()
    logger.info("run_created", run_id=run_id, construct=construct_name, mode=mode)
    return run_id


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str = "done",
    total_revisions: int = 0,
) -> None:
    """Mark a run as finished."""
    conn.execute(
        """UPDATE runs SET status = ?, total_revisions = ?, finished_at = ?
           WHERE id = ?""",
        (status, total_revisions, _now(), run_id),
    )
    conn.commit()
    logger.info("run_finished", run_id=run_id, status=status, revisions=total_revisions)


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


def save_research(
    conn: sqlite3.Connection,
    run_id: str,
    research_summary: str,
) -> None:
    """Save web research summary for a run."""
    conn.execute(
        "INSERT INTO research (run_id, research_summary, created_at) VALUES (?, ?, ?)",
        (run_id, research_summary, _now()),
    )
    conn.commit()


def get_cached_research(
    conn: sqlite3.Connection,
    construct_fingerprint: str,
    ttl_hours: int = 24,
) -> str | None:
    """Fetch most recent research_summary for this exact construct.

    Returns None if no research exists or if it's older than ttl_hours.
    Joins through runs table to match on construct_fingerprint.
    """
    row = conn.execute(
        """SELECT res.research_summary, res.created_at
           FROM research res
           JOIN runs r ON res.run_id = r.id
           WHERE r.construct_fingerprint = ?
           ORDER BY res.created_at DESC
           LIMIT 1""",
        (construct_fingerprint,),
    ).fetchone()

    if row is None:
        return None

    created = datetime.fromisoformat(row["created_at"])
    age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
    if age_hours > ttl_hours:
        return None

    return row["research_summary"]


# ---------------------------------------------------------------------------
# Generation Rounds
# ---------------------------------------------------------------------------


def save_generation_round(
    conn: sqlite3.Connection,
    run_id: str,
    round_number: int,
    phase: str,
    items_text: str,
) -> int:
    """Save a generation/revision round. Returns the round_id."""
    cursor = conn.execute(
        """INSERT INTO generation_rounds (run_id, round_number, phase, items_text, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, round_number, phase, items_text, _now()),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def get_latest_round_id(conn: sqlite3.Connection, run_id: str) -> int | None:
    """Get the most recent generation round ID for a run."""
    row = conn.execute(
        "SELECT id FROM generation_rounds WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return row["id"] if row else None


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------


def save_review(
    conn: sqlite3.Connection,
    round_id: int,
    content_review: str = "",
    linguistic_review: str = "",
    bias_review: str = "",
    meta_review: str = "",
) -> None:
    """Save review results for a generation round."""
    conn.execute(
        """INSERT INTO reviews (round_id, content_review, linguistic_review,
           bias_review, meta_review, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (round_id, content_review, linguistic_review, bias_review, meta_review, _now()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


def save_feedback(
    conn: sqlite3.Connection,
    round_id: int,
    source: str,
    feedback_text: str,
    decision: str,
) -> None:
    """Save human or LewMod feedback for a generation round."""
    conn.execute(
        """INSERT INTO feedback (round_id, source, feedback_text, decision, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (round_id, source, feedback_text, decision, _now()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Anti-Homogeneity: Read Previous Items
# ---------------------------------------------------------------------------


def get_previous_items(
    conn: sqlite3.Connection,
    construct_fingerprint: str,
    exclude_run_id: str | None = None,
    limit: int = 5,
) -> list[str]:
    """Fetch items_text from completed runs with the same construct fingerprint.

    Returns up to `limit` most recent final-round items from prior runs.
    Used by the Item Writer to avoid generating similar items across runs.

    Filters by construct_fingerprint (SHA-256 hash of the full construct
    definition) to ensure memory only returns items from runs that used
    the exact same construct — not just the same name.
    """
    query = """
        SELECT gr.items_text
        FROM generation_rounds gr
        JOIN runs r ON gr.run_id = r.id
        WHERE r.construct_fingerprint = ?
          AND r.status = 'done'
          AND gr.round_number = (
              SELECT MAX(gr2.round_number)
              FROM generation_rounds gr2
              WHERE gr2.run_id = gr.run_id
          )
    """
    params: list = [construct_fingerprint]

    if exclude_run_id:
        query += " AND r.id != ?"
        params.append(exclude_run_id)

    query += " ORDER BY r.finished_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [row["items_text"] for row in rows]
