"""Bias Reviewer Agent: evaluates demographic and cultural fairness.

Evaluates items for potential bias using natural language output.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts.templates import BIAS_REVIEWER_SYSTEM, BIAS_REVIEWER_TASK
from src.schemas.agent_outputs import BiasReviewerOutput
from src.schemas.state import ReviewChainState
from src.utils.console import format_structured_agent_output, print_agent_message
from src.utils.structured_output import invoke_structured_with_fix

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

    prompt = BIAS_REVIEWER_TASK.format(
        items_text=items_text,
        construct_name=construct_name,
        target_population=target_population,
    ) + (
        "\n\nReturn ONLY JSON with fields:\n"
        '{"items":[{"item_number":1,"score":5,"feedback":"..."}],"overall_summary":"..."}'
    )

    messages = [
        SystemMessage(content=BIAS_REVIEWER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    parsed = await invoke_structured_with_fix(
        agent_name="bias_reviewer",
        messages=messages,
        schema=BiasReviewerOutput,
    )
    review_text = json.dumps(parsed.model_dump(), ensure_ascii=True, indent=2)

    logger.info("bias_reviewer_done")

    print_agent_message("BiasReviewer", "Critic", format_structured_agent_output("BiasReviewer", parsed))

    return {"bias_review": review_text}
