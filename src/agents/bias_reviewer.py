"""Bias Reviewer Agent: evaluates demographic and cultural fairness.

Evaluates items for potential bias using natural language output.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.models import create_llm
from src.prompts.templates import BIAS_REVIEWER_SYSTEM, BIAS_REVIEWER_TASK
from src.schemas.state import ReviewChainState
from src.utils.console import print_agent_message, validate_llm_response

logger = structlog.get_logger(__name__)


async def bias_reviewer_node(state: ReviewChainState) -> dict:
    """Evaluate all items for demographic bias."""
    items_text = state.get("items_text", "")
    construct_name = state.get("construct_name", "")
    construct_definition = state.get("construct_definition", "")

    # Derive target population from construct definition so the bias reviewer
    # does not penalise domain-appropriate language (e.g. "work" for AAAW).
    target_population = (
        f"The target population for the \"{construct_name}\" construct is "
        f"defined by: {construct_definition}. "
        "Evaluate bias only with respect to this target population."
    ) if construct_definition else "General adult population."

    logger.info("bias_reviewer_start")

    llm = create_llm("bias_reviewer")

    prompt = BIAS_REVIEWER_TASK.format(
        items_text=items_text,
        construct_name=construct_name,
        target_population=target_population,
    )

    messages = [
        SystemMessage(content=BIAS_REVIEWER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    review_text = validate_llm_response(response.content, "BiasReviewer")

    logger.info("bias_reviewer_done")

    print_agent_message("BiasReviewer", "Critic", review_text)

    return {"bias_review": review_text}
