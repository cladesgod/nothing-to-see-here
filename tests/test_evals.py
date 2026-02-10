"""Tests for evaluation pipeline — dataset, config, score parsing, caching."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.config import AgentSettings
from src.evals.dataset import GOLDEN_ITEMS, get_golden_dataset, parse_items_from_text
from src.evals.evaluators import _parse_score


# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------


class TestGoldenDataset:
    """Test the golden dataset from paper Table 3."""

    def test_has_6_dimensions(self):
        assert len(GOLDEN_ITEMS) == 6

    def test_has_4_items_per_dimension(self):
        for dim_name, items in GOLDEN_ITEMS.items():
            assert len(items) == 4, f"{dim_name} should have 4 items"

    def test_total_24_items(self):
        assert sum(len(items) for items in GOLDEN_ITEMS.values()) == 24

    def test_dimensions_match_construct(self):
        from src.schemas.constructs import AAAW_CONSTRUCT
        assert set(GOLDEN_ITEMS.keys()) == {d.name for d in AAAW_CONSTRUCT.dimensions}

    def test_get_golden_dataset_returns_24_examples(self):
        assert len(get_golden_dataset()) == 24

    def test_example_has_required_fields(self):
        for ex in get_golden_dataset():
            assert "item_text" in ex
            assert "dimension_name" in ex
            assert "dimension_definition" in ex
            assert "orbiting_dimensions" in ex
            assert len(ex["orbiting_dimensions"]) == 2

    def test_orbiting_dimensions_are_tuples(self):
        for ex in get_golden_dataset():
            for orb in ex["orbiting_dimensions"]:
                assert isinstance(orb, tuple)
                assert len(orb) == 2


# ---------------------------------------------------------------------------
# Score Parsing
# ---------------------------------------------------------------------------


class TestScoreParsing:
    """Test the _parse_score helper."""

    def test_standard_format(self):
        assert _parse_score("SCORE: 0.85\nREASONING: Good.")["score"] == 0.85

    def test_fraction_format(self):
        assert abs(_parse_score("SCORE: 6/7\nREASONING: Strong.")["score"] - 6 / 7) < 0.01

    def test_clamps_above_1(self):
        assert _parse_score("SCORE: 1.5\nREASONING: x")["score"] == 1.0

    def test_clamps_below_0(self):
        assert _parse_score("SCORE: -0.3\nREASONING: x")["score"] == 0.0

    def test_no_score_returns_zero(self):
        assert _parse_score("This item is well-constructed.")["score"] == 0.0

    def test_with_extra_text(self):
        assert _parse_score("Analysis:\nSCORE: 0.92\nREASONING: Excellent.")["score"] == 0.92


# ---------------------------------------------------------------------------
# Item Text Parsing
# ---------------------------------------------------------------------------


class TestItemParsing:
    """Test parsing numbered items from text."""

    def test_numbered_items(self):
        text = "1. First.\n2. Second.\n3. Third."
        items = parse_items_from_text(text)
        assert len(items) == 3
        assert items[0] == "First."

    def test_parenthesis_format(self):
        assert len(parse_items_from_text("1) First.\n2) Second.")) == 2

    def test_empty_text(self):
        assert parse_items_from_text("") == []

    def test_ignores_non_numbered_lines(self):
        text = "Here are items:\n1. First.\nNote.\n2. Second."
        assert len(parse_items_from_text(text)) == 2


# ---------------------------------------------------------------------------
# Web Search Caching
# ---------------------------------------------------------------------------


class TestWebSearchCaching:
    """Test web search cache read/write."""

    def test_write_and_read(self):
        import src.agents.web_surfer as ws
        original_dir = ws.CACHE_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ws.CACHE_DIR = Path(tmpdir) / "web_search"
                ws._write_cache("Test", "query", "results here")
                assert ws._read_cache("Test", "query", ttl_hours=24) == "results here"
        finally:
            ws.CACHE_DIR = original_dir

    def test_cache_miss_returns_none(self):
        import src.agents.web_surfer as ws
        original_dir = ws.CACHE_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ws.CACHE_DIR = Path(tmpdir) / "web_search"
                assert ws._read_cache("NonExistent", "query", ttl_hours=24) is None
        finally:
            ws.CACHE_DIR = original_dir

    def test_expired_cache_returns_none(self):
        import src.agents.web_surfer as ws
        original_dir = ws.CACHE_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ws.CACHE_DIR = Path(tmpdir) / "web_search"
                ws.CACHE_DIR.mkdir(parents=True)
                path = ws._cache_path("Test", "query")
                path.write_text(json.dumps({
                    "query": "query",
                    "results": "old",
                    "timestamp": "2020-01-01T00:00:00+00:00",
                }))
                assert ws._read_cache("Test", "query", ttl_hours=24) is None
        finally:
            ws.CACHE_DIR = original_dir

    def test_corrupt_cache_is_removed(self):
        """FIX 4: Corrupt cache files should be deleted."""
        import src.agents.web_surfer as ws
        original_dir = ws.CACHE_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ws.CACHE_DIR = Path(tmpdir) / "web_search"
                ws.CACHE_DIR.mkdir(parents=True)
                path = ws._cache_path("Test", "query")
                path.write_text("NOT VALID JSON {{{{")
                assert ws._read_cache("Test", "query", ttl_hours=24) is None
                assert not path.exists(), "Corrupt cache file should be deleted"
        finally:
            ws.CACHE_DIR = original_dir

    def test_websurfer_config_has_cache_fields(self):
        s = AgentSettings()
        assert s.agents.websurfer.cache_enabled is True
        assert s.agents.websurfer.cache_ttl_hours == 24


# ---------------------------------------------------------------------------
# Eval Config
# ---------------------------------------------------------------------------


class TestEvalConfig:
    """Test eval configuration from agents.toml."""

    def test_eval_defaults(self):
        s = AgentSettings()
        assert s.eval.enabled is True
        assert s.eval.judge_temperature == 0.0
        assert s.eval.content_validity_threshold == 0.83
        assert s.eval.distinctiveness_threshold == 0.35

    def test_eval_from_dict(self):
        data = {"eval": {"judge_model": "gpt-4", "judge_temperature": 0.1, "content_validity_threshold": 0.90}}
        s = AgentSettings.model_validate(data)
        assert s.eval.judge_model == "gpt-4"
        assert s.eval.content_validity_threshold == 0.90
        assert s.eval.bias_threshold == 0.9  # default kept

    def test_agents_toml_has_eval_config(self):
        import tomllib
        toml_path = Path(__file__).parent.parent / "agents.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        s = AgentSettings.model_validate(data)
        assert s.eval.enabled is True
        assert s.eval.judge_model == "meta-llama/llama-4-maverick"


# ---------------------------------------------------------------------------
# Runner Key Consistency (FIX 1)
# ---------------------------------------------------------------------------


class TestRunnerKeyConsistency:
    """FIX 1: Ensure runner uses correct dict keys from get_golden_dataset()."""

    def test_golden_example_keys_match_runner_access(self):
        required_keys = {"item_text", "dimension_name", "dimension_definition", "orbiting_dimensions"}
        for ex in get_golden_dataset():
            assert required_keys.issubset(ex.keys()), f"Missing keys: {required_keys - ex.keys()}"
