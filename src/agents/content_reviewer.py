"""Content Reviewer Agent: evaluates content validity of items.

Uses the Colquitt et al. (2019) method: rates each item's relevance to
the target dimension and two orbiting (related) dimensions on a 1-7 scale.
Outputs natural language text with a markdown rating table.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.models import create_llm
from src.prompts.templates import CONTENT_REVIEWER_SYSTEM, CONTENT_REVIEWER_TASK
from src.schemas.state import ReviewChainState
from src.utils.console import print_agent_message, validate_llm_response

logger = structlog.get_logger(__name__)


async def content_reviewer_node(state: ReviewChainState) -> dict:
    """Evaluate all items' content validity using the Colquitt method."""
    items_text = state.get("items_text", "")
    dimension_info = state.get("dimension_info", "Not specified.")

    logger.info("content_reviewer_start")

    llm = create_llm("content_reviewer")

    prompt = CONTENT_REVIEWER_TASK.format(
        items_text=items_text,
        dimension_info=dimension_info,
    )

    messages = [
        SystemMessage(content=CONTENT_REVIEWER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    review_text = validate_llm_response(response.content, "ContentReviewer")

    logger.info("content_reviewer_done")

    print_agent_message("ContentReviewer", "Critic", review_text)

    return {"content_review": review_text}
