"""Tests for agent functions (unit tests with mocked LLM)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.critic import critic_node, critic_router
from src.agents.lewmod import lewmod_node
from src.config import AgentSettings
from src.schemas.agent_outputs import LewModOutput
from src.schemas.state import MainState
from src.utils.console import validate_llm_response


# ---------------------------------------------------------------------------
# Critic Router
# ---------------------------------------------------------------------------


class TestCriticRouter:
    """Tests for the deterministic critic routing function."""

    def test_web_research_routes_to_web_surfer(self):
        assert critic_router({"current_phase": "web_research"}) == "web_surfer"

    def test_item_generation_routes_to_item_writer(self):
        assert critic_router({"current_phase": "item_generation"}) == "item_writer"

    def test_review_routes_to_review_chain(self):
        assert critic_router({"current_phase": "review"}) == "review_chain"

    def test_human_feedback_routes_to_human_feedback(self):
        assert critic_router({"current_phase": "human_feedback"}) == "human_feedback"

    def test_revision_routes_to_item_writer(self):
        assert critic_router({"current_phase": "revision"}) == "item_writer"

    def test_done_routes_to_done(self):
        assert critic_router({"current_phase": "done"}) == "done"

    def test_empty_state_defaults_to_web_surfer(self):
        assert critic_router({}) == "web_surfer"


# ---------------------------------------------------------------------------
# Critic Node
# ---------------------------------------------------------------------------


class TestCriticNode:
    """Tests for the critic node."""

    def test_transitions_to_done_on_max_revisions(self):
        state: MainState = {
            "current_phase": "revision",
            "revision_count": 3,
            "max_revisions": 3,
        }
        result = critic_node(state)
        assert result["current_phase"] == "done"

    def test_passes_through_for_normal_phase(self):
        result = critic_node({"current_phase": "item_generation"})
        assert "current_phase" not in result
        assert "messages" in result


# ---------------------------------------------------------------------------
# LewMod Node
# ---------------------------------------------------------------------------


class TestLewModNode:
    """Tests for the LewMod automated feedback agent."""

    @pytest.mark.asyncio
    async def test_approves_on_decision_approve(self):
        mock_response = AsyncMock()
        mock_response.content = (
            '{"decision":"APPROVE","feedback":"Items are ready.","keep":[1,2],"revise":[],"discard":[]}'
        )

        with patch("src.agents.lewmod.invoke_structured_with_fix") as mock_invoke:
            mock_invoke.return_value = LewModOutput.model_validate_json(mock_response.content)

            state: MainState = {
                "items_text": "1. Test item.",
                "review_text": "All pass.",
                "revision_count": 2,
            }
            result = await lewmod_node(state)

            assert result["current_phase"] == "done"
            assert "Items are ready." in result["human_feedback"]

    @pytest.mark.asyncio
    async def test_revises_on_decision_revise(self):
        mock_response = AsyncMock()
        mock_response.content = (
            '{"decision":"REVISE","feedback":"Item 3 needs work.","keep":[1,2],"revise":[3],"discard":[]}'
        )

        with patch("src.agents.lewmod.invoke_structured_with_fix") as mock_invoke:
            mock_invoke.return_value = LewModOutput.model_validate_json(mock_response.content)

            state: MainState = {
                "items_text": "1. A\n2. B\n3. C",
                "review_text": "Item 3 low c-value.",
                "revision_count": 0,
            }
            result = await lewmod_node(state)

            assert result["current_phase"] == "revision"
            assert result["revision_count"] == 1
            assert result["human_item_decisions"] == {"1": "KEEP", "2": "KEEP", "3": "REVISE"}

    @pytest.mark.asyncio
    async def test_increments_revision_count(self):
        mock_response = AsyncMock()
        mock_response.content = (
            '{"decision":"REVISE","feedback":"Feedback.","keep":[],"revise":[1],"discard":[]}'
        )

        with patch("src.agents.lewmod.invoke_structured_with_fix") as mock_invoke:
            mock_invoke.return_value = LewModOutput.model_validate_json(mock_response.content)

            result = await lewmod_node({
                "items_text": "items",
                "review_text": "review",
                "revision_count": 2,
            })
            assert result["revision_count"] == 3

    @pytest.mark.asyncio
    async def test_discard_mapped_to_revise_for_writer(self):
        mock_response = AsyncMock()
        mock_response.content = (
            '{"decision":"REVISE","feedback":"Drop 4.","keep":[1],"revise":[2],"discard":[4]}'
        )
        with patch("src.agents.lewmod.invoke_structured_with_fix") as mock_invoke:
            mock_invoke.return_value = LewModOutput.model_validate_json(mock_response.content)
            result = await lewmod_node(
                {
                    "items_text": "items",
                    "review_text": "review",
                    "revision_count": 0,
                }
            )
        assert result["human_item_decisions"] == {"1": "KEEP", "2": "REVISE", "4": "REVISE"}

    @pytest.mark.asyncio
    async def test_approves_with_long_preamble(self):
        mock_response = AsyncMock()
        mock_response.content = (
            '{"decision":"APPROVE","feedback":"Ready.","keep":[1,2,3],"revise":[],"discard":[]}'
        )

        with patch("src.agents.lewmod.invoke_structured_with_fix") as mock_invoke:
            mock_invoke.return_value = LewModOutput.model_validate_json(mock_response.content)

            result = await lewmod_node({
                "items_text": "1. Test.",
                "review_text": "All pass.",
                "revision_count": 3,
            })
            assert result["current_phase"] == "done"

    @pytest.mark.asyncio
    async def test_filters_decisions_to_active_items(self):
        mock_response = AsyncMock()
        mock_response.content = (
            '{"decision":"REVISE","feedback":"Work on 1 only.","keep":[2],"revise":[1,8],"discard":[9]}'
        )
        with patch("src.agents.lewmod.invoke_structured_with_fix") as mock_invoke:
            mock_invoke.return_value = LewModOutput.model_validate_json(mock_response.content)
            result = await lewmod_node(
                {
                    "items_text": "1. A\n2. B\n8. C\n9. D",
                    "active_items_text": "1. A\n8. C",
                    "review_text": "review",
                    "revision_count": 1,
                }
            )
        assert result["human_item_decisions"] == {"1": "REVISE", "8": "REVISE"}

    @pytest.mark.asyncio
    async def test_auto_approves_when_no_active_items(self):
        with patch("src.agents.lewmod.invoke_structured_with_fix") as mock_invoke:
            result = await lewmod_node(
                {
                    "items_text": "1. A\n2. B",
                    "active_items_text": "",
                    "review_text": "review",
                    "revision_count": 2,
                }
            )
            mock_invoke.assert_not_called()
        assert result["current_phase"] == "done"
        assert result["human_item_decisions"] == {}


# ---------------------------------------------------------------------------
# LLM Response Validation (FIX 6)
# ---------------------------------------------------------------------------


class TestValidateLlmResponse:
    """Tests for the validate_llm_response helper."""

    def test_raises_on_none(self):
        with pytest.raises(ValueError, match="empty response"):
            validate_llm_response(None, "TestAgent")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="empty response"):
            validate_llm_response("", "TestAgent")

    def test_raises_on_whitespace_only(self):
        with pytest.raises(ValueError, match="empty response"):
            validate_llm_response("   \n\t  ", "TestAgent")

    def test_returns_stripped_content(self):
        assert validate_llm_response("  hello world  ", "Test") == "hello world"

    def test_passes_valid_content(self):
        text = "This is a valid response."
        assert validate_llm_response(text, "Test") == text

    def test_error_message_includes_agent_name(self):
        with pytest.raises(ValueError, match="WebSurfer"):
            validate_llm_response("", "WebSurfer")


# ---------------------------------------------------------------------------
# Web Search Caching
# ---------------------------------------------------------------------------


class TestWebSearchCaching:
    """Test web search file cache (fingerprint-based keys)."""

    FINGERPRINT = "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"

    def test_write_and_read(self):
        import src.agents.web_surfer as ws
        original_dir = ws.CACHE_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ws.CACHE_DIR = Path(tmpdir) / "web_search"
                ws._write_cache(self.FINGERPRINT, "query", "results here")
                assert ws._read_cache(self.FINGERPRINT, "query", ttl_hours=24) == "results here"
        finally:
            ws.CACHE_DIR = original_dir

    def test_cache_miss_returns_none(self):
        import src.agents.web_surfer as ws
        original_dir = ws.CACHE_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ws.CACHE_DIR = Path(tmpdir) / "web_search"
                assert ws._read_cache(self.FINGERPRINT, "query", ttl_hours=24) is None
        finally:
            ws.CACHE_DIR = original_dir

    def test_expired_cache_returns_none(self):
        import src.agents.web_surfer as ws
        original_dir = ws.CACHE_DIR
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ws.CACHE_DIR = Path(tmpdir) / "web_search"
                ws.CACHE_DIR.mkdir(parents=True)
                path = ws._cache_path(self.FINGERPRINT, "query")
                path.write_text(json.dumps({
                    "query": "query",
                    "results": "old",
                    "timestamp": "2020-01-01T00:00:00+00:00",
                }))
                assert ws._read_cache(self.FINGERPRINT, "query", ttl_hours=24) is None
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
                path = ws._cache_path(self.FINGERPRINT, "query")
                path.write_text("NOT VALID JSON {{{{")
                assert ws._read_cache(self.FINGERPRINT, "query", ttl_hours=24) is None
                assert not path.exists(), "Corrupt cache file should be deleted"
        finally:
            ws.CACHE_DIR = original_dir

    def test_different_fingerprints_different_cache_files(self):
        """Different constructs should not share cache entries."""
        import src.agents.web_surfer as ws
        fp1 = "aaaa" * 16
        fp2 = "bbbb" * 16
        path1 = ws._cache_path(fp1, "same query")
        path2 = ws._cache_path(fp2, "same query")
        assert path1 != path2

    def test_websurfer_config_has_cache_fields(self):
        s = AgentSettings()
        assert s.agents.websurfer.cache_enabled is True
        assert s.agents.websurfer.cache_ttl_hours == 24
