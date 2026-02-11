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
    get_cached_research,
    get_latest_round_id,
    get_previous_items,
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
        "construct_fingerprint": "test-fingerprint-aaaw",
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
        expected = {"runs", "research", "generation_rounds", "reviews", "feedback"}
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
# Anti-Homogeneity: get_previous_items
# ---------------------------------------------------------------------------


class TestGetPreviousItems:
    """Tests for the cross-run item diversity query (fingerprint-based)."""

    FINGERPRINT_AAAW = "fp-aaaw-abc123"
    FINGERPRINT_OTHER = "fp-bigfive-xyz789"

    def _setup_completed_run(self, db_conn, run_id, fingerprint, items_text, num_rounds=1):
        """Helper: create a completed run with generation rounds."""
        _create_test_run(db_conn, run_id, construct_fingerprint=fingerprint)
        for i in range(num_rounds):
            save_generation_round(db_conn, run_id, i, "generation" if i == 0 else "revision", items_text if i == num_rounds - 1 else "old items")
        finish_run(db_conn, run_id, status="done", total_revisions=num_rounds - 1)

    def test_returns_items_from_completed_runs(self, db_conn):
        self._setup_completed_run(db_conn, "done-1", self.FINGERPRINT_AAAW, "1. Item A\n2. Item B")
        result = get_previous_items(db_conn, self.FINGERPRINT_AAAW)
        assert len(result) == 1
        assert "Item A" in result[0]

    def test_excludes_current_run(self, db_conn):
        self._setup_completed_run(db_conn, "done-2", self.FINGERPRINT_AAAW, "1. Old items")
        _create_test_run(db_conn, "current", construct_fingerprint=self.FINGERPRINT_AAAW)
        save_generation_round(db_conn, "current", 0, "generation", "Current items")

        result = get_previous_items(db_conn, self.FINGERPRINT_AAAW, exclude_run_id="current")
        assert len(result) == 1
        assert "Old items" in result[0]

    def test_only_returns_matching_fingerprint(self, db_conn):
        self._setup_completed_run(db_conn, "aaaw-run", self.FINGERPRINT_AAAW, "AAAW items")
        self._setup_completed_run(db_conn, "other-run", self.FINGERPRINT_OTHER, "Big Five items")

        result = get_previous_items(db_conn, self.FINGERPRINT_AAAW)
        assert len(result) == 1
        assert "AAAW items" in result[0]

    def test_fingerprint_mismatch_returns_empty(self, db_conn):
        """Different fingerprint (even same construct name) → no memory."""
        self._setup_completed_run(db_conn, "run-v1", "fp-version-1", "V1 items")
        result = get_previous_items(db_conn, "fp-version-2")
        assert result == []

    def test_returns_only_final_round_items(self, db_conn):
        self._setup_completed_run(db_conn, "multi", self.FINGERPRINT_AAAW, "Final revised items", num_rounds=3)
        result = get_previous_items(db_conn, self.FINGERPRINT_AAAW)
        assert len(result) == 1
        assert "Final revised" in result[0]

    def test_respects_limit(self, db_conn):
        for i in range(10):
            self._setup_completed_run(db_conn, f"run-{i}", self.FINGERPRINT_AAAW, f"Items set {i}")
        result = get_previous_items(db_conn, self.FINGERPRINT_AAAW, limit=3)
        assert len(result) == 3

    def test_returns_empty_for_no_completed_runs(self, db_conn):
        _create_test_run(db_conn, "running", construct_fingerprint=self.FINGERPRINT_AAAW)
        save_generation_round(db_conn, "running", 0, "generation", "In progress items")
        result = get_previous_items(db_conn, self.FINGERPRINT_AAAW)
        assert result == []

    def test_excludes_failed_runs(self, db_conn):
        _create_test_run(db_conn, "failed-run", construct_fingerprint=self.FINGERPRINT_AAAW)
        save_generation_round(db_conn, "failed-run", 0, "generation", "Failed items")
        finish_run(db_conn, "failed-run", status="failed")

        result = get_previous_items(db_conn, self.FINGERPRINT_AAAW)
        assert result == []

    def test_null_fingerprint_not_matched(self, db_conn):
        """Runs without fingerprint (legacy) should not be returned."""
        # Manually insert a run without fingerprint
        db_conn.execute(
            """INSERT INTO runs (id, construct_name, construct_definition,
               construct_fingerprint, mode, model, max_revisions, status, started_at)
               VALUES (?, ?, ?, NULL, ?, ?, ?, 'done', datetime('now'))""",
            ("legacy-run", "AAAW", "Attitudes", "human", "maverick", 3),
        )
        db_conn.commit()
        save_generation_round(db_conn, "legacy-run", 0, "generation", "Legacy items")

        result = get_previous_items(db_conn, self.FINGERPRINT_AAAW)
        assert result == []


# ---------------------------------------------------------------------------
# Research Cache: get_cached_research
# ---------------------------------------------------------------------------


class TestGetCachedResearch:
    """Tests for fingerprint-based research summary caching."""

    FINGERPRINT_A = "fp-research-aaa"
    FINGERPRINT_B = "fp-research-bbb"

    def test_returns_research_for_matching_fingerprint(self, db_conn):
        _create_test_run(db_conn, "run-r1", construct_fingerprint=self.FINGERPRINT_A)
        save_research(db_conn, "run-r1", "Summary about AAAW.")
        finish_run(db_conn, "run-r1", status="done")

        result = get_cached_research(db_conn, self.FINGERPRINT_A, ttl_hours=24)
        assert result == "Summary about AAAW."

    def test_returns_none_for_different_fingerprint(self, db_conn):
        _create_test_run(db_conn, "run-r2", construct_fingerprint=self.FINGERPRINT_A)
        save_research(db_conn, "run-r2", "Summary A.")
        finish_run(db_conn, "run-r2", status="done")

        result = get_cached_research(db_conn, self.FINGERPRINT_B, ttl_hours=24)
        assert result is None

    def test_returns_none_when_no_research_exists(self, db_conn):
        result = get_cached_research(db_conn, self.FINGERPRINT_A, ttl_hours=24)
        assert result is None

    def test_returns_most_recent(self, db_conn):
        _create_test_run(db_conn, "run-old", construct_fingerprint=self.FINGERPRINT_A)
        save_research(db_conn, "run-old", "Old summary.")
        finish_run(db_conn, "run-old", status="done")

        _create_test_run(db_conn, "run-new", construct_fingerprint=self.FINGERPRINT_A)
        save_research(db_conn, "run-new", "New summary.")
        finish_run(db_conn, "run-new", status="done")

        result = get_cached_research(db_conn, self.FINGERPRINT_A, ttl_hours=24)
        assert result == "New summary."

    def test_respects_ttl(self, db_conn):
        """Research older than TTL should not be returned."""
        _create_test_run(db_conn, "run-ttl", construct_fingerprint=self.FINGERPRINT_A)
        # Insert research with old timestamp
        db_conn.execute(
            "INSERT INTO research (run_id, research_summary, created_at) VALUES (?, ?, ?)",
            ("run-ttl", "Very old summary.", "2020-01-01T00:00:00+00:00"),
        )
        db_conn.commit()

        result = get_cached_research(db_conn, self.FINGERPRINT_A, ttl_hours=24)
        assert result is None


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


class TestKeepItemLocking:
    """Test KEEP-lock behavior in Item Writer revision mode."""

    @staticmethod
    def _original_items() -> str:
        return (
            "1. I feel confident using AI tools at work.\n"
            "2. AI helps me complete my tasks faster.\n"
            "3. I worry AI might replace my job."
        )

    def test_parse_numbered_blocks(self):
        from src.agents.item_writer import _parse_numbered_blocks

        blocks = _parse_numbered_blocks(self._original_items())
        assert blocks[1].startswith("I feel confident")
        assert blocks[2].startswith("AI helps")
        assert blocks[3].startswith("I worry")

    def test_parse_numbered_blocks_ignores_response_scale_tail(self):
        from src.agents.item_writer import _parse_numbered_blocks

        text = self._original_items() + "\n\nResponse scale: 1 to 7"
        blocks = _parse_numbered_blocks(text)
        assert len(blocks) == 3
        assert "Response scale" not in blocks[3]

    def test_enforce_keep_restores_modified_keep_items(self):
        from src.agents.item_writer import _enforce_keep_locks

        generated = (
            "1. I am very confident with AI tools in every task.\n"
            "2. AI helps me complete my tasks faster.\n"
            "3. I worry AI might replace my job."
        )
        locked = _enforce_keep_locks(self._original_items(), generated, keep_numbers=[1, 3])
        assert "1. I feel confident using AI tools at work." in locked
        assert "2. AI helps me complete my tasks faster." in locked
        assert "3. I worry AI might replace my job." in locked

    def test_format_locked_items_includes_requested_keep_ids(self):
        from src.agents.item_writer import _format_locked_items

        text = _format_locked_items(self._original_items(), [1, 3])
        assert "Locked KEEP Items" in text
        assert "Item 1" in text
        assert "Item 3" in text

    @pytest.mark.asyncio
    async def test_extract_keep_numbers_from_meta_json_block(self):
        from src.agents.item_writer import _extract_keep_numbers

        review_text = """
```json
{
  "items": [
    {"item_number": 1, "decision": "KEEP", "reason": "ok", "revised_item_stem": null},
    {"item_number": 2, "decision": "REVISE", "reason": "clarity", "revised_item_stem": "AI improves task speed."},
    {"item_number": 3, "decision": "KEEP", "reason": "ok", "revised_item_stem": null}
  ],
  "overall_synthesis": "good set"
}
```
"""
        keep = await _extract_keep_numbers(review_text)
        assert keep == [1, 3]

    def test_parse_human_directives_keep_and_revise(self):
        from src.agents.item_writer import _parse_human_directives

        keep, revise = _parse_human_directives("KEEP: 1, 3, 5\nREVISE: 2,4")
        assert keep == [1, 3, 5]
        assert revise == [2, 4]

    def test_parse_human_directives_revise_overrides_keep(self):
        from src.agents.item_writer import _parse_human_directives

        keep, revise = _parse_human_directives("KEEP: 1,2,3\nREVISE: 2")
        assert keep == [1, 3]
        assert revise == [2]

    def test_get_human_decisions_prefers_structured_state(self):
        from src.agents.item_writer import _get_human_decisions

        keep, revise = _get_human_decisions(
            {
                "human_item_decisions": {
                    1: "KEEP",
                    "2": "REVISE",
                    "99": "KEEP",
                    "x": "KEEP",
                },
                "human_feedback": "KEEP: 2",
            },
            valid_item_numbers={1, 2, 3},
        )
        assert keep == [1]
        assert revise == [2]

    def test_get_human_decisions_falls_back_to_legacy_feedback(self):
        from src.agents.item_writer import _get_human_decisions

        keep, revise = _get_human_decisions(
            {
                "human_item_decisions": {},
                "human_feedback": "KEEP: 1,3\nREVISE: 3",
            },
            valid_item_numbers={1, 2, 3},
        )
        assert keep == [1]
        assert revise == [3]

    def test_align_generated_to_targets_maps_by_order_when_renumbered(self):
        from src.agents.item_writer import _align_generated_to_targets

        generated = {1: "Revised A", 2: "Revised B"}
        aligned = _align_generated_to_targets(generated, target_numbers=[2, 4])
        assert aligned == {2: "Revised A", 4: "Revised B"}

    def test_align_generated_to_targets_prefers_exact_target_keys(self):
        from src.agents.item_writer import _align_generated_to_targets

        generated = {
            1: "Keep one",
            2: "Keep two",
            3: "Revise three",
            4: "Revise four",
            8: "Revise eight",
        }
        aligned = _align_generated_to_targets(generated, target_numbers=[3, 4, 8])
        assert aligned == {3: "Revise three", 4: "Revise four", 8: "Revise eight"}
