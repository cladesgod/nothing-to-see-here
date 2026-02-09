"""Main Workflow Graph: orchestrates the full item generation pipeline.

Matches the paper's architecture (Fig. 1 & Fig. 2):

                    ┌──→ web_surfer ──┐
                    │                 │
  START → critic ──┼──→ item_writer ──┼──→ critic ──→ END (output)
            ↑       │                 │       ↑
            │       ├──→ review_chain ─┤       │
            │       │   (subgraph)    │       │
            │       └──→ human_feedback┘       │
            │                                  │
            └──────────────────────────────────┘

The Critic Agent is the central hub / supervisor.
Every worker node feeds back into the Critic, which decides the next step.

All agent communication uses natural language text (paper-like style).
"""

from __future__ import annotations

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy, interrupt

from src.agents.critic import critic_node, critic_router
from src.config import get_agent_settings
from src.agents.item_writer import item_writer_node
from src.agents.lewmod import lewmod_node
from src.agents.web_surfer import web_surfer_node
from src.graphs.review_chain import review_chain_graph
from src.schemas.constructs import AAAW_CONSTRUCT
from src.schemas.state import MainState, ReviewChainState
from src.utils.console import print_agent_message, print_human_prompt

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Review chain wrapper: runs the subgraph for all items as a batch
# ---------------------------------------------------------------------------

async def review_chain_wrapper(state: MainState) -> dict:
    """Run the review chain subgraph for the batch of items."""
    items_text = state.get("items_text", "")
    construct_name = state.get("construct_name", "")
    construct_definition = state.get("construct_definition", "")

    logger.info("review_chain_start")

    # Build dimension info text for the content reviewer
    # Include all dimensions with their orbiting constructs
    dimension_lines = []
    for dim in AAAW_CONSTRUCT.dimensions:
        orbiting = AAAW_CONSTRUCT.get_orbiting_definitions(dim.name)
        dim_text = f"**Construct 1 (TARGET): {dim.name}**\n- Definition: {dim.definition}"
        if len(orbiting) >= 2:
            dim_text += f"\n**Construct 2 (ORBITING): {orbiting[0][0]}**\n- Definition: {orbiting[0][1]}"
            dim_text += f"\n**Construct 3 (ORBITING): {orbiting[1][0]}**\n- Definition: {orbiting[1][1]}"
        dimension_lines.append(dim_text)

    dimension_info = "\n\n".join(dimension_lines)

    # Run the review chain subgraph with all items at once
    sub_state: ReviewChainState = {
        "items_text": items_text,
        "construct_name": construct_name,
        "construct_definition": construct_definition,
        "dimension_info": dimension_info,
    }

    result = await review_chain_graph.ainvoke(sub_state)
    review_text = result.get("meta_review", "No review available.")

    logger.info("review_chain_done")

    return {
        "review_text": review_text,
        "current_phase": "human_feedback",
        "messages": ["[ReviewChain] Review completed"],
    }


# ---------------------------------------------------------------------------
# Human feedback node: uses interrupt() for human-in-the-loop
# ---------------------------------------------------------------------------

def human_feedback_node(state: MainState) -> dict:
    """Collect human feedback via interrupt.

    Pauses the graph execution and waits for human input.
    Resume with: graph.invoke(Command(resume="your feedback here"), config)
    """
    items_text = state.get("items_text", "")
    review_text = state.get("review_text", "")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    # Build a summary for the human reviewer
    summary = (
        f"## Generated Items\n\n{items_text}\n\n"
        f"---\n\n"
        f"## Meta Editor Review\n\n{review_text}\n\n"
        f"---\n\n"
        f"**Revision round:** {revision_count + 1}/{max_revisions}\n\n"
        f"Please provide your feedback (or type 'approve' to accept all):"
    )

    print_human_prompt(summary)

    # Interrupt: pause and wait for human input
    human_input = interrupt(summary)

    # Process the human response
    if isinstance(human_input, str) and human_input.strip().lower() == "approve":
        print_agent_message("Human", "Critic", "Approved all items.")
        return {
            "human_feedback": "approved",
            "current_phase": "done",
            "messages": ["[Human] Approved all items"],
        }

    feedback = human_input if isinstance(human_input, str) else str(human_input)
    print_agent_message("Human", "Critic", feedback)

    return {
        "human_feedback": feedback,
        "current_phase": "revision",
        "revision_count": revision_count + 1,
        "messages": [f"[Human] Provided feedback for revision round {revision_count + 1}"],
    }


# ---------------------------------------------------------------------------
# Build the main workflow graph
# ---------------------------------------------------------------------------

_CRITIC_EDGES = {
    "web_surfer": "web_surfer",
    "item_writer": "item_writer",
    "review_chain": "review_chain",
    "human_feedback": "human_feedback",
    "done": END,
}


def build_main_workflow(checkpointer=None, lewmod=False):
    """Build and compile the main workflow graph.

    Architecture (matching paper Fig. 1 & 2):
      - Critic Agent is a visible central node
      - All worker nodes → Critic → conditional routing
      - Review chain is a subgraph with parallel reviewers

    Args:
        checkpointer: LangGraph checkpointer for persistence.
                      - None  → use MemorySaver (default, for CLI / standalone)
                      - False → no checkpointer (for LangGraph Platform)
                      - Any Checkpointer instance → use that directly
        lewmod: If True, use LewMod (automated LLM feedback) instead of
                human-in-the-loop feedback. Default: False.
    """
    builder = StateGraph(MainState)

    # ---- Retry policy for LLM nodes (configurable via agents.toml) ----
    agent_settings = get_agent_settings()
    retry = RetryPolicy(
        max_attempts=agent_settings.retry.max_attempts,
        initial_interval=agent_settings.retry.initial_interval,
        backoff_factor=agent_settings.retry.backoff_factor,
    )

    # ---- Nodes ----
    builder.add_node("critic", critic_node)  # Deterministic — no retry
    builder.add_node("web_surfer", web_surfer_node, retry_policy=retry)
    builder.add_node("item_writer", item_writer_node, retry_policy=retry)
    builder.add_node("review_chain", review_chain_wrapper, retry_policy=retry)
    if lewmod:
        builder.add_node("human_feedback", lewmod_node, retry_policy=retry)
    else:
        builder.add_node("human_feedback", human_feedback_node)  # Interrupt — no retry

    # ---- Edges ----
    # START → Critic (always enters through critic first)
    builder.add_edge(START, "critic")

    # Critic → conditional routing to workers or END
    builder.add_conditional_edges("critic", critic_router, _CRITIC_EDGES)

    # All workers → back to Critic (the central hub pattern from Fig. 1)
    builder.add_edge("web_surfer", "critic")
    builder.add_edge("item_writer", "critic")
    builder.add_edge("review_chain", "critic")
    builder.add_edge("human_feedback", "critic")

    # ---- Compile ----
    if checkpointer is None:
        checkpointer = MemorySaver()
    elif checkpointer is False:
        checkpointer = None  # LangGraph Platform provides its own

    return builder.compile(checkpointer=checkpointer)


# Module-level graph instance for `langgraph dev` / LangGraph Studio.
# Referenced in langgraph.json as "aig_workflow": "./src/graphs/main_workflow.py:graph"
# NOTE: No checkpointer here — LangGraph Platform provides its own persistence.
graph = build_main_workflow(checkpointer=False)
