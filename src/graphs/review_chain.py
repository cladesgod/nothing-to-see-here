"""Review Chain Subgraph: parallel reviewers → meta editor.

Based on the paper's diagram (Fig. 2):
  Content Reviewer ─┐
  Linguistic Reviewer ──→ Meta Editor
  Bias Reviewer ─────┘

The three reviewers run in PARALLEL (fan-out), then their results
converge at the Meta Editor (fan-in).

All communication is natural language text (paper-like style).
"""

from __future__ import annotations

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy

from src.agents.bias_reviewer import bias_reviewer_node
from src.agents.content_reviewer import content_reviewer_node
from src.agents.linguistic_reviewer import linguistic_reviewer_node
from src.agents.meta_editor import meta_editor_node
from src.schemas.state import ReviewChainState


def build_review_chain() -> StateGraph:
    """Build the review chain subgraph.

    Fan-out:  START → [content, linguistic, bias]  (parallel)
    Fan-in:   [content, linguistic, bias] → meta_editor → END
    """
    from src.config import get_agent_settings

    builder = StateGraph(ReviewChainState)

    # Retry policy for reviewer nodes (configurable via agents.toml)
    agent_settings = get_agent_settings()
    retry = RetryPolicy(
        max_attempts=agent_settings.retry.max_attempts,
        initial_interval=agent_settings.retry.initial_interval,
        backoff_factor=agent_settings.retry.backoff_factor,
    )

    # Add nodes with retry policy
    builder.add_node("content_reviewer", content_reviewer_node, retry_policy=retry)
    builder.add_node("linguistic_reviewer", linguistic_reviewer_node, retry_policy=retry)
    builder.add_node("bias_reviewer", bias_reviewer_node, retry_policy=retry)
    builder.add_node("meta_editor", meta_editor_node, retry_policy=retry)

    # Fan-out: START → all three reviewers in parallel
    builder.add_edge(START, "content_reviewer")
    builder.add_edge(START, "linguistic_reviewer")
    builder.add_edge(START, "bias_reviewer")

    # Fan-in: all three reviewers → meta_editor
    builder.add_edge("content_reviewer", "meta_editor")
    builder.add_edge("linguistic_reviewer", "meta_editor")
    builder.add_edge("bias_reviewer", "meta_editor")

    # meta_editor → END
    builder.add_edge("meta_editor", END)

    return builder


# Pre-built compiled subgraph (without checkpointer - parent handles it)
review_chain_graph = build_review_chain().compile()
