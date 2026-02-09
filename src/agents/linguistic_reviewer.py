"""Linguistic Reviewer Agent: evaluates grammar, readability, and clarity.

Evaluates items on linguistic criteria using natural language output.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.models import create_llm
from src.prompts.templates import LINGUISTIC_REVIEWER_SYSTEM, LINGUISTIC_REVIEWER_TASK
from src.schemas.state import ReviewChainState
from src.utils.console import print_agent_message

logger = structlog.get_logger(__name__)


async def linguistic_reviewer_node(state: ReviewChainState) -> dict:
    """Evaluate all items' linguistic quality."""
    items_text = state.get("items_text", "")
    construct_name = state.get("construct_name", "")

    logger.info("linguistic_reviewer_start")

    llm = create_llm("linguistic_reviewer")

    prompt = LINGUISTIC_REVIEWER_TASK.format(
        items_text=items_text,
        construct_name=construct_name,
    )

    messages = [
        SystemMessage(content=LINGUISTIC_REVIEWER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    review_text = response.content

    logger.info("linguistic_reviewer_done")

    print_agent_message("LinguisticReviewer", "Critic", review_text)

    return {"linguistic_review": review_text}
