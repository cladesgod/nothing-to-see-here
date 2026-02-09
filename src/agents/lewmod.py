"""LewMod Agent: automated LLM-based feedback replacing human-in-the-loop.

Persona: Senior psychometrician with expertise in construct validity,
scale development, and item quality assessment.
Uses temperature=0.3 for balanced, expert-like evaluation.

Activated via `python run.py --lewmod`.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.models import create_llm
from src.prompts.templates import LEWMOD_SYSTEM, LEWMOD_TASK
from src.schemas.state import MainState
from src.utils.console import print_agent_message

logger = structlog.get_logger(__name__)


async def lewmod_node(state: MainState) -> dict:
    """LewMod agent node: provides automated expert feedback or approval.

    Reads items_text and review_text from state, evaluates holistically,
    and either approves (current_phase='done') or requests revision
    (current_phase='revision') with specific feedback.
    """
    items_text = state.get("items_text", "")
    review_text = state.get("review_text", "")
    revision_count = state.get("revision_count", 0)

    logger.info("lewmod_start", revision_count=revision_count)

    llm = create_llm("lewmod")

    prompt = LEWMOD_TASK.format(
        items_text=items_text,
        review_text=review_text,
        revision_count=revision_count,
    )

    messages = [
        SystemMessage(content=LEWMOD_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    feedback_text = response.content

    logger.info("lewmod_done", revision_count=revision_count)

    print_agent_message("LewMod", "Critic", feedback_text)

    # Parse the decision from the response
    # LewMod is instructed to start with "DECISION: APPROVE" or "DECISION: REVISE"
    lower_text = feedback_text.strip().lower()
    if "decision: approve" in lower_text[:200]:
        return {
            "human_feedback": feedback_text,
            "current_phase": "done",
            "messages": [f"[LewMod] Approved items after {revision_count} revision(s)"],
        }

    return {
        "human_feedback": feedback_text,
        "current_phase": "revision",
        "revision_count": revision_count + 1,
        "messages": [f"[LewMod] Revision requested (round {revision_count + 1})"],
    }
