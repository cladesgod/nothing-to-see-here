"""Tests for workflow graph structure and retry policies."""

from __future__ import annotations

from src.graphs.main_workflow import build_main_workflow
from src.graphs.review_chain import build_review_chain


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
