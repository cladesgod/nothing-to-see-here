"""Critic Agent: central orchestrator of the workflow.

In the paper (Fig. 1), the Critic Agent is the central hub:
  - Item Writer <-> Critic <-> Review Chain
  - Human Feedback -> Critic -> Output

Implemented as a visible graph node that sets routing via current_phase,
plus a deterministic routing function for conditional edges.
"""

from __future__ import annotations

from typing import Literal

import structlog

from src.schemas.state import MainState
from src.utils.console import print_phase_transition

logger = structlog.get_logger(__name__)

# Type alias for the routing destinations
Route = Literal[
    "web_surfer",
    "item_writer",
    "review_chain",
    "human_feedback",
    "done",
]


def critic_node(state: MainState) -> dict:
    """Critic Agent node: logs the decision and passes through.

    The actual routing decision is made by critic_router (conditional edge).
    This node makes the Critic visible in the graph visualization and
    provides a hook for future LLM-based critic logic.
    """
    current_phase = state.get("current_phase", "web_research")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    logger.info(
        "critic_agent",
        phase=current_phase,
        revision_count=revision_count,
    )

    # If revision phase and max reached, transition to done
    if current_phase == "revision" and revision_count >= max_revisions:
        logger.info("critic_max_revisions_reached", count=revision_count)
        print_phase_transition("done")
        return {
            "current_phase": "done",
            "messages": [f"[Critic] Max revisions ({max_revisions}) reached. Finalizing."],
        }

    print_phase_transition(current_phase)

    return {
        "messages": [f"[Critic] Routing → {current_phase}"],
    }


def critic_router(state: MainState) -> Route:
    """Deterministic router that decides the next workflow step.

    Used as the conditional-edge function AFTER the critic node.

    Routing logic based on current_phase:
      web_research    → web_surfer
      item_generation → item_writer
      review          → review_chain
      human_feedback  → human_feedback
      revision        → item_writer
      done            → END
    """
    current_phase = state.get("current_phase", "web_research")

    if current_phase == "web_research":
        return "web_surfer"

    if current_phase == "item_generation":
        return "item_writer"

    if current_phase == "review":
        return "review_chain"

    if current_phase == "human_feedback":
        return "human_feedback"

    if current_phase == "revision":
        return "item_writer"

    # done or unknown
    return "done"
