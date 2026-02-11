"""Tests for workflow graph structure and retry policies."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.graphs.main_workflow import review_chain_wrapper
from src.graphs.main_workflow import human_feedback_node
from src.graphs.main_workflow import build_main_workflow
from src.graphs.review_chain import build_review_chain
from src.schemas.phases import Phase


class TestReviewChainGraph:
    """Tests for the review chain subgraph structure."""

    def test_has_correct_nodes(self):
        graph = build_review_chain().compile()
        nodes = set(graph.get_graph().nodes.keys())
        expected = {"__start__", "__end__", "content_reviewer", "linguistic_reviewer", "bias_reviewer", "meta_editor"}
        assert nodes == expected

    def test_reviewers_fan_into_meta_editor(self):
        mermaid = build_review_chain().compile().get_graph().draw_mermaid()
        for name in ["content_reviewer", "linguistic_reviewer", "bias_reviewer", "meta_editor"]:
            assert name in mermaid


class TestMainWorkflow:
    """Tests for the main workflow graph structure."""

    def test_has_correct_nodes(self):
        nodes = set(build_main_workflow().get_graph().nodes.keys())
        expected = {"__start__", "__end__", "critic", "web_surfer", "item_writer", "review_chain", "human_feedback"}
        assert nodes == expected

    def test_critic_is_central_hub(self):
        mermaid = build_main_workflow().get_graph().draw_mermaid()
        assert "critic" in mermaid
        for node in ["web_surfer", "item_writer", "review_chain", "human_feedback"]:
            assert node in mermaid

    def test_compiles_successfully(self):
        assert build_main_workflow() is not None


class TestMainWorkflowWithLewMod:
    """Tests for LewMod-enabled workflow."""

    def test_has_same_node_names(self):
        nodes = set(build_main_workflow(lewmod=True).get_graph().nodes.keys())
        expected = {"__start__", "__end__", "critic", "web_surfer", "item_writer", "review_chain", "human_feedback"}
        assert nodes == expected

    def test_compiles_successfully(self):
        assert build_main_workflow(lewmod=True) is not None


class TestRetryPolicy:
    """Tests for retry policy on graph nodes."""

    def test_llm_nodes_have_retry(self):
        graph = build_main_workflow()
        for name in ["web_surfer", "item_writer", "review_chain"]:
            assert graph.builder.nodes[name].retry_policy is not None, f"{name} missing retry"

    def test_critic_has_no_retry(self):
        assert build_main_workflow().builder.nodes["critic"].retry_policy is None

    def test_human_feedback_no_retry_in_human_mode(self):
        assert build_main_workflow(lewmod=False).builder.nodes["human_feedback"].retry_policy is None

    def test_lewmod_feedback_has_retry(self):
        assert build_main_workflow(lewmod=True).builder.nodes["human_feedback"].retry_policy is not None

    def test_review_chain_nodes_have_retry(self):
        nodes = build_review_chain().nodes
        for name in ["content_reviewer", "linguistic_reviewer", "bias_reviewer", "meta_editor"]:
            assert nodes[name].retry_policy is not None, f"{name} missing retry"

    def test_retry_values_match_config(self):
        from src.config import get_agent_settings
        cfg = get_agent_settings().retry
        policy = build_main_workflow().builder.nodes["web_surfer"].retry_policy
        assert policy.max_attempts == cfg.max_attempts
        assert policy.initial_interval == cfg.initial_interval
        assert policy.backoff_factor == cfg.backoff_factor


class TestHumanFeedbackNode:
    """Tests for structured human feedback payload handling."""

    @pytest.mark.asyncio
    async def test_accepts_structured_payload_and_routes_to_revision(self):
        payload = {
            "approve": False,
            "item_decisions": {"1": "KEEP", "2": "REVISE"},
            "global_note": "Please improve item 2 clarity.",
        }
        with (
            patch("src.graphs.main_workflow.interrupt", return_value=payload),
            patch("src.utils.injection_defense.check_prompt_injection", return_value=(True, "")),
        ):
            result = await human_feedback_node(
                {
                    "items_text": "1. A\n2. B",
                    "review_text": '{"items":[{"item_number":1,"decision":"KEEP","reason":"","revised_item_stem":null}]}',
                    "revision_count": 0,
                    "max_revisions": 3,
                }
            )
        assert result["current_phase"] == Phase.REVISION
        assert result["revision_count"] == 1
        assert result["human_item_decisions"] == {"1": "KEEP", "2": "REVISE"}
        assert result["human_global_note"] == "Please improve item 2 clarity."

    @pytest.mark.asyncio
    async def test_approve_payload_routes_to_done(self):
        payload = {"approve": True, "item_decisions": {}, "global_note": ""}
        with patch("src.graphs.main_workflow.interrupt", return_value=payload):
            result = await human_feedback_node(
                {
                    "items_text": "1. A",
                    "review_text": "{}",
                    "revision_count": 1,
                    "max_revisions": 3,
                }
            )
        assert result["current_phase"] == Phase.DONE
        assert result["human_feedback"] == "approved"

    @pytest.mark.asyncio
    async def test_human_feedback_panel_formats_meta_review_summary(self):
        payload = {"approve": True, "item_decisions": {}, "global_note": ""}
        review_text = (
            '{"items":[{"item_number":1,"decision":"KEEP","reason":"ok","revised_item_stem":null}],'
            '"overall_synthesis":"x"}'
        )
        with patch("src.graphs.main_workflow.interrupt", return_value=payload) as mock_interrupt:
            await human_feedback_node(
                {
                    "items_text": "1. A",
                    "review_text": review_text,
                    "revision_count": 0,
                    "max_revisions": 3,
                }
            )
        summary = mock_interrupt.call_args[0][0]
        assert "Decisions: KEEP=1, REVISE=0, DISCARD=0" in summary

    @pytest.mark.asyncio
    async def test_human_feedback_panel_uses_active_items_and_lists_frozen(self):
        payload = {"approve": True, "item_decisions": {}, "global_note": ""}
        with patch("src.graphs.main_workflow.interrupt", return_value=payload) as mock_interrupt:
            await human_feedback_node(
                {
                    "items_text": "1. A\n2. B\n3. C",
                    "active_items_text": "1. A\n3. C",
                    "frozen_item_numbers": [2],
                    "review_text": '{"items":[],"overall_synthesis":"x"}',
                    "revision_count": 1,
                    "max_revisions": 3,
                }
            )
        summary = mock_interrupt.call_args[0][0]
        assert "## Active Items for Review" in summary
        assert "1. A\n3. C" in summary
        assert "**Frozen KEEP items (auto-kept):** 2" in summary

    @pytest.mark.asyncio
    async def test_review_chain_skip_uses_overall_synthesis_key(self):
        result = await review_chain_wrapper(
            {
                "items_text": "1. Item",
                "active_items_text": "",
                "construct_name": "AAAW",
                "construct_definition": "Def",
                "dimension_info": "Info",
            }
        )
        assert '"overall_synthesis"' in result["review_text"]
        assert '"global_synthesis"' not in result["review_text"]
