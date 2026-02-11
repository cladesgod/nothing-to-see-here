"""Content Reviewer Agent: evaluates content validity of items.

Uses the Colquitt et al. (2019) method: rates each item's relevance to
the target dimension and two orbiting (related) dimensions on a 1-7 scale.
Outputs natural language text with a markdown rating table.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts.templates import CONTENT_REVIEWER_SYSTEM, CONTENT_REVIEWER_TASK
from src.schemas.agent_outputs import ContentReviewerOutput
from src.schemas.state import ReviewChainState
from src.utils.console import format_structured_agent_output, print_agent_message
from src.utils.structured_output import invoke_structured_with_fix

logger = structlog.get_logger(__name__)


async def content_reviewer_node(state: ReviewChainState) -> dict:
    """Evaluate all items' content validity using the Colquitt method."""
    items_text = state.get("items_text", "")
    dimension_info = state.get("dimension_info", "Not specified.")

    logger.info("content_reviewer_start")

    prompt = CONTENT_REVIEWER_TASK.format(
        items_text=items_text,
        dimension_info=dimension_info,
    ) + (
        "\n\nReturn ONLY JSON with fields:\n"
        '{"items":[{"item_number":1,"target_rating":6,"orbiting_1_rating":3,'
        '"orbiting_2_rating":2,"feedback":"..."}],"overall_summary":"..."}'
    )

    messages = [
        SystemMessage(content=CONTENT_REVIEWER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    parsed = await invoke_structured_with_fix(
        agent_name="content_reviewer",
        messages=messages,
        schema=ContentReviewerOutput,
    )
    review_text = json.dumps(parsed.model_dump(), ensure_ascii=True, indent=2)

    logger.info("content_reviewer_done")

    print_agent_message("ContentReviewer", "Critic", format_structured_agent_output("ContentReviewer", parsed))

    return {"content_review": review_text}
