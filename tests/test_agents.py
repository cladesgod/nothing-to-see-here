"""Tests for agent functions (unit tests with mocked LLM)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.critic import critic_node, critic_router
from src.agents.lewmod import lewmod_node
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
        mock_response.content = "DECISION: APPROVE\n\nItems are ready."

        with patch("src.agents.lewmod.create_llm") as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            state: MainState = {
                "items_text": "1. Test item.",
                "review_text": "All pass.",
                "revision_count": 2,
            }
            result = await lewmod_node(state)

            assert result["current_phase"] == "done"
            assert "DECISION: APPROVE" in result["human_feedback"]

    @pytest.mark.asyncio
    async def test_revises_on_decision_revise(self):
        mock_response = AsyncMock()
        mock_response.content = "DECISION: REVISE\n\nItem 3 needs work."

        with patch("src.agents.lewmod.create_llm") as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            state: MainState = {
                "items_text": "1. Test item.",
                "review_text": "Item 3 low c-value.",
                "revision_count": 0,
            }
            result = await lewmod_node(state)

            assert result["current_phase"] == "revision"
            assert result["revision_count"] == 1

    @pytest.mark.asyncio
    async def test_increments_revision_count(self):
        mock_response = AsyncMock()
        mock_response.content = "DECISION: REVISE\n\nFeedback."

        with patch("src.agents.lewmod.create_llm") as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await lewmod_node({
                "items_text": "items",
                "review_text": "review",
                "revision_count": 2,
            })
            assert result["revision_count"] == 3

    @pytest.mark.asyncio
    async def test_approves_with_long_preamble(self):
        """FIX 3: LewMod should detect APPROVE even after >200 chars preamble."""
        preamble = "After careful analysis of all items, " * 10  # >200 chars
        mock_response = AsyncMock()
        mock_response.content = preamble + "\nDECISION: APPROVE\n\nReady."

        with patch("src.agents.lewmod.create_llm") as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await lewmod_node({
                "items_text": "1. Test.",
                "review_text": "All pass.",
                "revision_count": 3,
            })
            assert result["current_phase"] == "done"


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
