"""Persistence layer for pipeline state.

Supports two backends:
- **SQLite** (default): Zero dependencies, used for CLI and development.
- **PostgreSQL** (production): Used when DATABASE_URL starts with "postgresql://".

The connection interface is unified — both backends return a connection
that supports execute(), fetchone(), fetchall(), commit(), close().
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DB_PATH = Path("data/pipeline.db")

# ---------------------------------------------------------------------------
# Schema (compatible with both SQLite and PostgreSQL)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    construct_name  TEXT NOT NULL,
    construct_definition TEXT,
    construct_fingerprint TEXT,
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

"""

# PostgreSQL version uses SERIAL instead of AUTOINCREMENT
PG_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    construct_name  TEXT NOT NULL,
    construct_definition TEXT,
    construct_fingerprint TEXT,
    mode            TEXT NOT NULL,
    model           TEXT,
    max_revisions   INTEGER,
    total_revisions INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'running',
    started_at      TEXT NOT NULL,
    finished_at     TEXT
);

CREATE TABLE IF NOT EXISTS research (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    research_summary TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_rounds (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    round_number    INTEGER NOT NULL,
    phase           TEXT NOT NULL,
    items_text      TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id              SERIAL PRIMARY KEY,
    round_id        INTEGER NOT NULL REFERENCES generation_rounds(id),
    content_review  TEXT,
    linguistic_review TEXT,
    bias_review     TEXT,
    meta_review     TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id              SERIAL PRIMARY KEY,
    round_id        INTEGER NOT NULL REFERENCES generation_rounds(id),
    source          TEXT NOT NULL,
    feedback_text   TEXT NOT NULL,
    decision        TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Get a database connection.

    Routing logic:
    - If db_path starts with "postgresql://", returns a psycopg connection.
    - Otherwise, returns a SQLite connection (default).

    For PostgreSQL, the psycopg package must be installed
    (included in the [api] optional dependency group).
    """
    path_str = str(db_path) if db_path else ""

    if path_str.startswith("postgresql://") or path_str.startswith("postgres://"):
        return _get_pg_connection(path_str)

    return _get_sqlite_connection(db_path)


def _get_sqlite_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Get or create SQLite connection. Auto-creates tables on first use."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_sqlite_tables(conn)
    return conn


def _get_pg_connection(db_url: str):
    """Get a PostgreSQL connection via psycopg.

    Returns a psycopg connection with dict row factory for
    compatibility with the SQLite Row interface.
    """
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        raise ImportError(
            "PostgreSQL support requires 'psycopg'. "
            "Install with: pip install -e '.[api]'"
        )

    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    _ensure_pg_tables(conn)
    logger.info("pg_connection_established", db_url=db_url[:30] + "...")
    return conn


# ---------------------------------------------------------------------------
# Table setup
# ---------------------------------------------------------------------------


def _ensure_sqlite_tables(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist. Runs migrations for schema changes."""
    conn.executescript(SCHEMA_SQL)
    # Migration: add construct_fingerprint column to existing DBs
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "construct_fingerprint" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN construct_fingerprint TEXT")
    conn.commit()


def _ensure_pg_tables(conn) -> None:
    """Create tables in PostgreSQL if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(PG_SCHEMA_SQL)
    logger.info("pg_tables_ensured")
