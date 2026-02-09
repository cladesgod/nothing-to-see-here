"""Bias Reviewer Agent: evaluates demographic and cultural fairness.

Evaluates items for potential bias using natural language output.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.models import create_llm
from src.prompts.templates import BIAS_REVIEWER_SYSTEM, BIAS_REVIEWER_TASK
from src.schemas.state import ReviewChainState
from src.utils.console import print_agent_message

logger = structlog.get_logger(__name__)


async def bias_reviewer_node(state: ReviewChainState) -> dict:
    """Evaluate all items for demographic bias."""
    items_text = state.get("items_text", "")
    construct_name = state.get("construct_name", "")

    logger.info("bias_reviewer_start")

    llm = create_llm("bias_reviewer")

    prompt = BIAS_REVIEWER_TASK.format(
        items_text=items_text,
        construct_name=construct_name,
    )

    messages = [
        SystemMessage(content=BIAS_REVIEWER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    review_text = response.content

    logger.info("bias_reviewer_done")

    print_agent_message("BiasReviewer", "Critic", review_text)

    return {"bias_review": review_text}
