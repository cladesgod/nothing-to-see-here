"""Linguistic Reviewer Agent: evaluates grammar, readability, and clarity.

Evaluates items on linguistic criteria using natural language output.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts.templates import LINGUISTIC_REVIEWER_SYSTEM, LINGUISTIC_REVIEWER_TASK
from src.schemas.agent_outputs import LinguisticReviewerOutput
from src.schemas.state import ReviewChainState
from src.utils.console import format_structured_agent_output, print_agent_message
from src.utils.structured_output import invoke_structured_with_fix

logger = structlog.get_logger(__name__)


async def linguistic_reviewer_node(state: ReviewChainState) -> dict:
    """Evaluate all items' linguistic quality."""
    items_text = state.get("items_text", "")
    construct_name = state.get("construct_name", "")

    logger.info("linguistic_reviewer_start")

    prompt = LINGUISTIC_REVIEWER_TASK.format(
        items_text=items_text,
        construct_name=construct_name,
    ) + (
        "\n\nReturn ONLY JSON with fields:\n"
        '{"items":[{"item_number":1,"grammatical_accuracy":5,'
        '"ease_of_understanding":5,"negative_language_free":4,'
        '"clarity_directness":4,"feedback":"..."}],"overall_summary":"..."}'
    )

    messages = [
        SystemMessage(content=LINGUISTIC_REVIEWER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    parsed = await invoke_structured_with_fix(
        agent_name="linguistic_reviewer",
        messages=messages,
        schema=LinguisticReviewerOutput,
    )
    review_text = json.dumps(parsed.model_dump(), ensure_ascii=True, indent=2)

    logger.info("linguistic_reviewer_done")

    print_agent_message(
        "LinguisticReviewer", "Critic", format_structured_agent_output("LinguisticReviewer", parsed)
    )

    return {"linguistic_review": review_text}
