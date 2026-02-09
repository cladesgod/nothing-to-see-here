"""Integration tests for the workflow graph structure."""

from __future__ import annotations

import pytest

from src.graphs.main_workflow import build_main_workflow
from src.graphs.review_chain import build_review_chain


class TestReviewChainGraph:
    """Tests for the review chain subgraph structure."""

    def test_review_chain_has_correct_nodes(self):
        builder = build_review_chain()
        graph = builder.compile()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "__start__",
            "__end__",
            "content_reviewer",
            "linguistic_reviewer",
            "bias_reviewer",
            "meta_editor",
        }
        assert expected == node_names

    def test_reviewers_fan_into_meta_editor(self):
        """Verify all three reviewers connect to meta_editor (parallel fan-in)."""
        builder = build_review_chain()
        graph = builder.compile()
        mermaid = graph.get_graph().draw_mermaid()
        # All three reviewers should have edges to meta_editor
        assert "content_reviewer" in mermaid
        assert "linguistic_reviewer" in mermaid
        assert "bias_reviewer" in mermaid
        assert "meta_editor" in mermaid


class TestMainWorkflow:
    """Tests for the main workflow graph structure."""

    def test_main_workflow_has_correct_nodes(self):
        graph = build_main_workflow()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "__start__",
            "__end__",
            "critic",
            "web_surfer",
            "item_writer",
            "review_chain",
            "human_feedback",
        }
        assert expected == node_names

    def test_critic_is_central_hub(self):
        """Verify critic connects to all worker nodes (paper Fig. 1 pattern)."""
        graph = build_main_workflow()
        mermaid = graph.get_graph().draw_mermaid()
        # START should go to critic
        assert "__start__" in mermaid
        assert "critic" in mermaid
        # All workers should appear in the graph
        for node in ["web_surfer", "item_writer", "review_chain", "human_feedback"]:
            assert node in mermaid

    def test_main_workflow_compiles(self):
        graph = build_main_workflow()
        assert graph is not None


class TestMainWorkflowWithLewMod:
    """Tests for the main workflow graph with LewMod enabled."""

    def test_lewmod_workflow_has_correct_nodes(self):
        """Graph should have the same node names regardless of lewmod flag."""
        graph = build_main_workflow(lewmod=True)
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "__start__",
            "__end__",
            "critic",
            "web_surfer",
            "item_writer",
            "review_chain",
            "human_feedback",  # Same node name, different function
        }
        assert expected == node_names

    def test_lewmod_workflow_compiles(self):
        graph = build_main_workflow(lewmod=True)
        assert graph is not None


class TestRetryPolicy:
    """Tests for retry policy on graph nodes."""

    def test_llm_nodes_have_retry_policy(self):
        """LLM-calling nodes should have a retry policy configured."""
        graph = build_main_workflow()
        nodes = graph.builder.nodes
        for name in ["web_surfer", "item_writer", "review_chain"]:
            assert nodes[name].retry_policy is not None, f"{name} missing retry_policy"

    def test_critic_has_no_retry_policy(self):
        """Critic is deterministic — no retry needed."""
        graph = build_main_workflow()
        assert graph.builder.nodes["critic"].retry_policy is None

    def test_human_feedback_no_retry_in_human_mode(self):
        """Human feedback uses interrupt — should not have retry."""
        graph = build_main_workflow(lewmod=False)
        assert graph.builder.nodes["human_feedback"].retry_policy is None

    def test_lewmod_feedback_has_retry(self):
        """LewMod feedback is LLM-based — should have retry."""
        graph = build_main_workflow(lewmod=True)
        assert graph.builder.nodes["human_feedback"].retry_policy is not None

    def test_review_chain_nodes_have_retry(self):
        """All review chain nodes should have retry policy."""
        builder = build_review_chain()
        nodes = builder.nodes
        for name in ["content_reviewer", "linguistic_reviewer", "bias_reviewer", "meta_editor"]:
            assert nodes[name].retry_policy is not None, f"{name} missing retry_policy"

    def test_retry_policy_matches_config(self):
        """Retry policy values should match agents.toml config."""
        from src.config import get_agent_settings
        cfg = get_agent_settings().retry
        graph = build_main_workflow()
        policy = graph.builder.nodes["web_surfer"].retry_policy
        assert policy.max_attempts == cfg.max_attempts
        assert policy.initial_interval == cfg.initial_interval
        assert policy.backoff_factor == cfg.backoff_factor
