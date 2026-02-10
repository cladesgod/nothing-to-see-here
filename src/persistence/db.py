"""SQLite persistence layer for pipeline state.

Stores full pipeline lifecycle: runs, research, items, reviews, feedback, evals.
Uses Python stdlib sqlite3 — zero additional dependencies.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DB_PATH = Path("data/pipeline.db")

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    construct_name  TEXT NOT NULL,
    construct_definition TEXT,
    mode            TEXT NOT NULL,
    model           TEXT,
    max_revisions   INTEGER,
    total_revisions INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'running',
    started_at      TEXT NOT NULL,
    finished_at     TEXT
);

CREATE TABLE IF NOT EXISTS research (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    research_summary TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_rounds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    round_number    INTEGER NOT NULL,
    phase           TEXT NOT NULL,
    items_text      TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id        INTEGER NOT NULL REFERENCES generation_rounds(id),
    content_review  TEXT,
    linguistic_review TEXT,
    bias_review     TEXT,
    meta_review     TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id        INTEGER NOT NULL REFERENCES generation_rounds(id),
    source          TEXT NOT NULL,
    feedback_text   TEXT NOT NULL,
    decision        TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    item_text       TEXT NOT NULL,
    dimension_name  TEXT,
    content_score   REAL,
    linguistic_score REAL,
    bias_score      REAL,
    overall_score   REAL,
    content_reasoning TEXT,
    linguistic_reasoning TEXT,
    bias_reasoning  TEXT,
    overall_reasoning TEXT,
    created_at      TEXT NOT NULL
);
"""


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Get or create SQLite connection. Auto-creates tables on first use."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()
