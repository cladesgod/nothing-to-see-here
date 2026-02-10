"""Meta Editor Agent: synthesizes all reviewer feedback into recommendations.

Integrates content, linguistic, and bias reviews to produce final
keep/revise/discard recommendations in natural language.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.models import create_llm
from src.prompts.templates import META_EDITOR_SYSTEM, META_EDITOR_TASK
from src.schemas.state import ReviewChainState
from src.utils.console import print_agent_message, validate_llm_response

logger = structlog.get_logger(__name__)


async def meta_editor_node(state: ReviewChainState) -> dict:
    """Synthesize all reviews and produce recommendations."""
    items_text = state.get("items_text", "")
    content_review = state.get("content_review", "No content review available.")
    linguistic_review = state.get("linguistic_review", "No linguistic review available.")
    bias_review = state.get("bias_review", "No bias review available.")

    logger.info("meta_editor_start")

    llm = create_llm("meta_editor")

    prompt = META_EDITOR_TASK.format(
        items_text=items_text,
        content_review=content_review,
        linguistic_review=linguistic_review,
        bias_review=bias_review,
    )

    messages = [
        SystemMessage(content=META_EDITOR_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    review_text = validate_llm_response(response.content, "MetaEditor")

    logger.info("meta_editor_done")

    print_agent_message("MetaEditor", "Critic", review_text)

    return {"meta_review": review_text}
