"""Tests for persistence layer (SQLite DB + repository functions)."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.persistence.db import SCHEMA_SQL, get_connection
from src.persistence.repository import (
    create_run,
    finish_run,
    get_latest_round_id,
    get_previous_items,
    save_eval_result,
    save_feedback,
    save_generation_round,
    save_research,
    save_review,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_conn(tmp_path: Path):
    """Create an in-memory-like temp DB with schema applied."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    yield conn
    conn.close()


def _create_test_run(conn: sqlite3.Connection, run_id: str = "run-1", **kwargs) -> str:
    """Shortcut to create a run with sensible defaults."""
    defaults = {
        "construct_name": "AAAW",
        "construct_definition": "Attitudes Toward AI",
        "mode": "lewmod",
        "model": "llama-4-maverick",
        "max_revisions": 3,
    }
    defaults.update(kwargs)
    return create_run(conn, run_id=run_id, **defaults)


# ---------------------------------------------------------------------------
# DB Connection & Schema
# ---------------------------------------------------------------------------


class TestDatabaseConnection:
    """Test SQLite connection and schema creation."""

    def test_creates_db_file(self, tmp_path: Path):
        db_path = tmp_path / "sub" / "pipeline.db"
        conn = get_connection(db_path)
        assert db_path.exists()
        conn.close()

    def test_row_factory_set(self, db_conn: sqlite3.Connection):
        assert db_conn.row_factory == sqlite3.Row

    def test_wal_mode_enabled(self, db_conn: sqlite3.Connection):
        mode = db_conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, db_conn: sqlite3.Connection):
        fk = db_conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_all_tables_created(self, db_conn: sqlite3.Connection):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {"runs", "research", "generation_rounds", "reviews", "feedback", "eval_results"}
        assert expected.issubset(tables)

    def test_idempotent_schema_creation(self, db_conn: sqlite3.Connection):
        """Running schema a second time should not error."""
        db_conn.executescript(SCHEMA_SQL)
        db_conn.commit()


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class TestRunsCRUD:
    """Test run creation and finalization."""

    def test_create_run(self, db_conn):
        run_id = _create_test_run(db_conn, "run-abc")
        assert run_id == "run-abc"
        row = db_conn.execute("SELECT * FROM runs WHERE id = ?", ("run-abc",)).fetchone()
        assert row["construct_name"] == "AAAW"
        assert row["status"] == "running"
        assert row["finished_at"] is None

    def test_finish_run(self, db_conn):
        _create_test_run(db_conn, "run-fin")
        finish_run(db_conn, "run-fin", status="done", total_revisions=2)
        row = db_conn.execute("SELECT * FROM runs WHERE id = ?", ("run-fin",)).fetchone()
        assert row["status"] == "done"
        assert row["total_revisions"] == 2
        assert row["finished_at"] is not None

    def test_finish_run_failed_status(self, db_conn):
        _create_test_run(db_conn, "run-fail")
        finish_run(db_conn, "run-fail", status="failed")
        row = db_conn.execute("SELECT * FROM runs WHERE id = ?", ("run-fail",)).fetchone()
        assert row["status"] == "failed"


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


class TestResearchCRUD:
    def test_save_research(self, db_conn):
        _create_test_run(db_conn, "run-r")
        save_research(db_conn, "run-r", "Summary of findings.")
        row = db_conn.execute("SELECT * FROM research WHERE run_id = ?", ("run-r",)).fetchone()
        assert row["research_summary"] == "Summary of findings."


# ---------------------------------------------------------------------------
# Generation Rounds
# ---------------------------------------------------------------------------


class TestGenerationRoundsCRUD:
    def test_save_generation_round_returns_id(self, db_conn):
        _create_test_run(db_conn, "run-g")
        round_id = save_generation_round(db_conn, "run-g", 0, "generation", "1. Item one\n2. Item two")
        assert isinstance(round_id, int)
        assert round_id > 0

    def test_multiple_rounds(self, db_conn):
        _create_test_run(db_conn, "run-g2")
        id1 = save_generation_round(db_conn, "run-g2", 0, "generation", "Round 0 items")
        id2 = save_generation_round(db_conn, "run-g2", 1, "revision", "Round 1 items")
        assert id2 > id1

    def test_get_latest_round_id(self, db_conn):
        _create_test_run(db_conn, "run-lr")
        save_generation_round(db_conn, "run-lr", 0, "generation", "Items")
        round_id_1 = get_latest_round_id(db_conn, "run-lr")
        save_generation_round(db_conn, "run-lr", 1, "revision", "Revised")
        round_id_2 = get_latest_round_id(db_conn, "run-lr")
        assert round_id_2 > round_id_1

    def test_get_latest_round_id_returns_none_if_empty(self, db_conn):
        _create_test_run(db_conn, "run-empty")
        assert get_latest_round_id(db_conn, "run-empty") is None


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------


class TestReviewsCRUD:
    def test_save_review(self, db_conn):
        _create_test_run(db_conn, "run-rev")
        round_id = save_generation_round(db_conn, "run-rev", 0, "generation", "Items")
        save_review(db_conn, round_id, content_review="Good", linguistic_review="OK", bias_review="Clean", meta_review="All pass")
        row = db_conn.execute("SELECT * FROM reviews WHERE round_id = ?", (round_id,)).fetchone()
        assert row["content_review"] == "Good"
        assert row["meta_review"] == "All pass"


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class TestFeedbackCRUD:
    def test_save_human_feedback(self, db_conn):
        _create_test_run(db_conn, "run-fb")
        round_id = save_generation_round(db_conn, "run-fb", 0, "generation", "Items")
        save_feedback(db_conn, round_id, source="human", feedback_text="Looks good", decision="approve")
        row = db_conn.execute("SELECT * FROM feedback WHERE round_id = ?", (round_id,)).fetchone()
        assert row["source"] == "human"
        assert row["decision"] == "approve"

    def test_save_lewmod_feedback(self, db_conn):
        _create_test_run(db_conn, "run-lm")
        round_id = save_generation_round(db_conn, "run-lm", 0, "generation", "Items")
        save_feedback(db_conn, round_id, source="lewmod", feedback_text="DECISION: REVISE", decision="revise")
        row = db_conn.execute("SELECT * FROM feedback WHERE round_id = ?", (round_id,)).fetchone()
        assert row["source"] == "lewmod"
        assert row["decision"] == "revise"


# ---------------------------------------------------------------------------
# Eval Results
# ---------------------------------------------------------------------------


class TestEvalResultsCRUD:
    def test_save_eval_result(self, db_conn):
        _create_test_run(db_conn, "run-ev")
        scores = {
            "content": {"score": 0.85, "reasoning": "Good"},
            "linguistic": {"score": 0.9, "reasoning": "Clear"},
            "bias": {"score": 0.95, "reasoning": "Unbiased"},
            "overall": {"score": 0.9, "reasoning": "Strong"},
        }
        save_eval_result(db_conn, "run-ev", "I feel anxious.", "AI Use Anxiety", scores)
        row = db_conn.execute("SELECT * FROM eval_results WHERE run_id = ?", ("run-ev",)).fetchone()
        assert row["content_score"] == 0.85
        assert row["overall_reasoning"] == "Strong"


# ---------------------------------------------------------------------------
# Anti-Homogeneity: get_previous_items
# ---------------------------------------------------------------------------


class TestGetPreviousItems:
    """Tests for the cross-run item diversity query."""

    def _setup_completed_run(self, db_conn, run_id, construct, items_text, num_rounds=1):
        """Helper: create a completed run with generation rounds."""
        _create_test_run(db_conn, run_id, construct_name=construct)
        for i in range(num_rounds):
            save_generation_round(db_conn, run_id, i, "generation" if i == 0 else "revision", items_text if i == num_rounds - 1 else "old items")
        finish_run(db_conn, run_id, status="done", total_revisions=num_rounds - 1)

    def test_returns_items_from_completed_runs(self, db_conn):
        self._setup_completed_run(db_conn, "done-1", "AAAW", "1. Item A\n2. Item B")
        result = get_previous_items(db_conn, "AAAW")
        assert len(result) == 1
        assert "Item A" in result[0]

    def test_excludes_current_run(self, db_conn):
        self._setup_completed_run(db_conn, "done-2", "AAAW", "1. Old items")
        _create_test_run(db_conn, "current", construct_name="AAAW")
        save_generation_round(db_conn, "current", 0, "generation", "Current items")

        result = get_previous_items(db_conn, "AAAW", exclude_run_id="current")
        assert len(result) == 1
        assert "Old items" in result[0]

    def test_only_returns_matching_construct(self, db_conn):
        self._setup_completed_run(db_conn, "aaaw-run", "AAAW", "AAAW items")
        self._setup_completed_run(db_conn, "other-run", "Big Five", "Big Five items")

        result = get_previous_items(db_conn, "AAAW")
        assert len(result) == 1
        assert "AAAW items" in result[0]

    def test_returns_only_final_round_items(self, db_conn):
        self._setup_completed_run(db_conn, "multi", "AAAW", "Final revised items", num_rounds=3)
        result = get_previous_items(db_conn, "AAAW")
        assert len(result) == 1
        assert "Final revised" in result[0]

    def test_respects_limit(self, db_conn):
        for i in range(10):
            self._setup_completed_run(db_conn, f"run-{i}", "AAAW", f"Items set {i}")
        result = get_previous_items(db_conn, "AAAW", limit=3)
        assert len(result) == 3

    def test_returns_empty_for_no_completed_runs(self, db_conn):
        _create_test_run(db_conn, "running")
        save_generation_round(db_conn, "running", 0, "generation", "In progress items")
        result = get_previous_items(db_conn, "AAAW")
        assert result == []

    def test_excludes_failed_runs(self, db_conn):
        _create_test_run(db_conn, "failed-run", construct_name="AAAW")
        save_generation_round(db_conn, "failed-run", 0, "generation", "Failed items")
        finish_run(db_conn, "failed-run", status="failed")

        result = get_previous_items(db_conn, "AAAW")
        assert result == []


# ---------------------------------------------------------------------------
# Item Writer Diversity Helper
# ---------------------------------------------------------------------------


class TestFormatItemHistory:
    """Test the _format_item_history helper in item_writer."""

    def test_empty_list_returns_empty_string(self):
        from src.agents.item_writer import _format_item_history
        assert _format_item_history([]) == ""

    def test_single_set(self):
        from src.agents.item_writer import _format_item_history
        result = _format_item_history(["1. Item A\n2. Item B"])
        assert "Prior Set 1" in result
        assert "Item A" in result
        assert "avoid similarity" in result.lower()

    def test_multiple_sets(self):
        from src.agents.item_writer import _format_item_history
        result = _format_item_history(["Set one", "Set two", "Set three"])
        assert "Prior Set 1" in result
        assert "Prior Set 2" in result
        assert "Prior Set 3" in result
