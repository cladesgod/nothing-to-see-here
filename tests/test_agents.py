"""Tests for agent functions (unit tests with mocked LLM)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.critic import critic_node, critic_router
from src.agents.lewmod import lewmod_node
from src.schemas.state import MainState


class TestCriticRouter:
    """Tests for the deterministic critic routing function."""

    def test_routes_to_web_surfer_on_web_research(self):
        state: MainState = {"current_phase": "web_research"}
        assert critic_router(state) == "web_surfer"

    def test_routes_to_item_writer_on_item_generation(self):
        state: MainState = {"current_phase": "item_generation"}
        assert critic_router(state) == "item_writer"

    def test_routes_to_review_chain_on_review(self):
        state: MainState = {"current_phase": "review"}
        assert critic_router(state) == "review_chain"

    def test_routes_to_human_feedback(self):
        state: MainState = {"current_phase": "human_feedback"}
        assert critic_router(state) == "human_feedback"

    def test_routes_to_item_writer_on_revision(self):
        state: MainState = {"current_phase": "revision", "revision_count": 1}
        assert critic_router(state) == "item_writer"

    def test_routes_to_done_on_done_phase(self):
        state: MainState = {"current_phase": "done"}
        assert critic_router(state) == "done"

    def test_defaults_to_web_research_on_empty_state(self):
        state: MainState = {}
        assert critic_router(state) == "web_surfer"


class TestCriticNode:
    """Tests for the critic node (visible in graph)."""

    def test_transitions_to_done_on_max_revisions(self):
        state: MainState = {
            "current_phase": "revision",
            "revision_count": 3,
            "max_revisions": 3,
        }
        result = critic_node(state)
        assert result["current_phase"] == "done"

    def test_passes_through_for_normal_phase(self):
        state: MainState = {"current_phase": "item_generation"}
        result = critic_node(state)
        assert "current_phase" not in result  # no override
        assert "messages" in result


class TestLewModNode:
    """Tests for the LewMod automated feedback agent."""

    @pytest.mark.asyncio
    async def test_lewmod_approves_when_decision_approve(self):
        """LewMod should set current_phase='done' when it outputs DECISION: APPROVE."""
        mock_response = AsyncMock()
        mock_response.content = "DECISION: APPROVE\n\nThe items are ready for pilot testing."

        with patch("src.agents.lewmod.create_llm") as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            state: MainState = {
                "items_text": "1. I feel anxious when using AI at work.",
                "review_text": "All items meet thresholds.",
                "revision_count": 2,
            }
            result = await lewmod_node(state)

            assert result["current_phase"] == "done"
            assert result["human_feedback"] == mock_response.content

    @pytest.mark.asyncio
    async def test_lewmod_revises_when_decision_revise(self):
        """LewMod should set current_phase='revision' when it outputs DECISION: REVISE."""
        mock_response = AsyncMock()
        mock_response.content = "DECISION: REVISE\n\nItem 3 needs rewording for clarity."

        with patch("src.agents.lewmod.create_llm") as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            state: MainState = {
                "items_text": "1. I feel anxious when using AI at work.",
                "review_text": "Item 3 has low c-value.",
                "revision_count": 0,
            }
            result = await lewmod_node(state)

            assert result["current_phase"] == "revision"
            assert result["revision_count"] == 1
            assert result["human_feedback"] == mock_response.content

    @pytest.mark.asyncio
    async def test_lewmod_increments_revision_count(self):
        """LewMod should correctly increment revision_count on revise."""
        mock_response = AsyncMock()
        mock_response.content = "DECISION: REVISE\n\nFeedback here."

        with patch("src.agents.lewmod.create_llm") as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            state: MainState = {
                "items_text": "items",
                "review_text": "review",
                "revision_count": 2,
            }
            result = await lewmod_node(state)

            assert result["revision_count"] == 3


class TestContentReviewerNode:
    """Tests for the content reviewer agent with calculator tool."""

    @pytest.mark.asyncio
    async def test_content_reviewer_without_calculator(self):
        """When calculator is disabled, standard LLM call (no tool loop)."""
        from src.agents.content_reviewer import content_reviewer_node
        from src.config import AgentSettings, ContentReviewerConfig

        mock_response = AsyncMock()
        mock_response.content = "Review: all items have strong content validity."
        mock_response.tool_calls = []

        mock_cfg = ContentReviewerConfig(calculator=False)

        with (
            patch("src.agents.content_reviewer.create_llm") as mock_create,
            patch("src.agents.content_reviewer.get_agent_settings") as mock_settings,
        ):
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            mock_agent_settings = MagicMock()
            mock_agent_settings.get_agent_config.return_value = mock_cfg
            mock_settings.return_value = mock_agent_settings

            state = {
                "items_text": "1. I feel anxious when using AI.",
                "dimension_info": "Target: AI Use Anxiety",
            }
            result = await content_reviewer_node(state)

            assert result["content_review"] == mock_response.content
            # create_llm should NOT be called with tools
            mock_create.assert_called_once_with("content_reviewer")

    @pytest.mark.asyncio
    async def test_content_reviewer_with_calculator_no_tool_calls(self):
        """When calculator enabled but LLM doesn't call tools, still works."""
        from src.agents.content_reviewer import content_reviewer_node
        from src.config import ContentReviewerConfig

        mock_response = AsyncMock()
        mock_response.content = "Review text with computed values."
        mock_response.tool_calls = []

        mock_cfg = ContentReviewerConfig(calculator=True)

        with (
            patch("src.agents.content_reviewer.create_llm") as mock_create,
            patch("src.agents.content_reviewer.get_agent_settings") as mock_settings,
        ):
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            mock_agent_settings = MagicMock()
            mock_agent_settings.get_agent_config.return_value = mock_cfg
            mock_settings.return_value = mock_agent_settings

            state = {
                "items_text": "1. I feel anxious when using AI.",
                "dimension_info": "Target: AI Use Anxiety",
            }
            result = await content_reviewer_node(state)

            assert result["content_review"] == mock_response.content
            # create_llm should be called WITH tools
            from src.tools.calculator import calculate

            mock_create.assert_called_once_with(
                "content_reviewer", tools=[calculate]
            )

    @pytest.mark.asyncio
    async def test_content_reviewer_handles_wrong_param_name(self):
        """When model sends wrong param name (e.g. 'c' instead of 'expression'),
        _extract_expression should recover and calculate correctly."""
        from src.agents.content_reviewer import content_reviewer_node
        from src.config import ContentReviewerConfig

        # First response: LLM returns tool_calls with wrong param name
        mock_response_with_tool = MagicMock()
        mock_response_with_tool.content = ""
        mock_response_with_tool.tool_calls = [
            {"id": "call_1", "name": "calculate", "args": {"c": "6/6"}},
        ]

        # Second response: LLM returns final text (no more tool calls)
        mock_final_response = MagicMock()
        mock_final_response.content = "c-value = 1.0, meets threshold."
        mock_final_response.tool_calls = []

        mock_cfg = ContentReviewerConfig(calculator=True)

        with (
            patch("src.agents.content_reviewer.create_llm") as mock_create,
            patch("src.agents.content_reviewer.get_agent_settings") as mock_settings,
        ):
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = [
                mock_response_with_tool,
                mock_final_response,
            ]
            mock_create.return_value = mock_llm

            mock_agent_settings = MagicMock()
            mock_agent_settings.get_agent_config.return_value = mock_cfg
            mock_settings.return_value = mock_agent_settings

            state = {
                "items_text": "1. I feel anxious when using AI.",
                "dimension_info": "Target: AI Use Anxiety",
            }
            result = await content_reviewer_node(state)

            assert result["content_review"] == "c-value = 1.0, meets threshold."
            # LLM should be called twice (initial + after tool result)
            assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_content_reviewer_tool_loop_with_correct_params(self):
        """When model sends correct param, tool loop works end-to-end."""
        from src.agents.content_reviewer import content_reviewer_node
        from src.config import ContentReviewerConfig

        mock_response_with_tool = MagicMock()
        mock_response_with_tool.content = ""
        mock_response_with_tool.tool_calls = [
            {"id": "call_1", "name": "calculate", "args": {"expression": "5/6"}},
            {"id": "call_2", "name": "calculate", "args": {"expression": "((6-2)+(6-3))/2/6"}},
        ]

        mock_final_response = MagicMock()
        mock_final_response.content = "Item 1: c=0.8333, d=0.5833"
        mock_final_response.tool_calls = []

        mock_cfg = ContentReviewerConfig(calculator=True)

        with (
            patch("src.agents.content_reviewer.create_llm") as mock_create,
            patch("src.agents.content_reviewer.get_agent_settings") as mock_settings,
        ):
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = [
                mock_response_with_tool,
                mock_final_response,
            ]
            mock_create.return_value = mock_llm

            mock_agent_settings = MagicMock()
            mock_agent_settings.get_agent_config.return_value = mock_cfg
            mock_settings.return_value = mock_agent_settings

            state = {
                "items_text": "1. I feel anxious when using AI.",
                "dimension_info": "Target: AI Use Anxiety",
            }
            result = await content_reviewer_node(state)

            assert result["content_review"] == "Item 1: c=0.8333, d=0.5833"
            assert mock_llm.ainvoke.call_count == 2
