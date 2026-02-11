"""Meta Editor Agent: synthesizes all reviewer feedback into recommendations.

Integrates content, linguistic, and bias reviews to produce final
keep/revise/discard recommendations in natural language.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts.templates import META_EDITOR_SYSTEM, META_EDITOR_TASK
from src.schemas.agent_outputs import MetaEditorOutput
from src.schemas.state import ReviewChainState
from src.utils.console import format_structured_agent_output, print_agent_message
from src.utils.structured_output import invoke_structured_with_fix

logger = structlog.get_logger(__name__)


async def meta_editor_node(state: ReviewChainState) -> dict:
    """Synthesize all reviews and produce recommendations."""
    items_text = state.get("items_text", "")
    content_review = state.get("content_review", "No content review available.")
    linguistic_review = state.get("linguistic_review", "No linguistic review available.")
    bias_review = state.get("bias_review", "No bias review available.")

    logger.info("meta_editor_start")

    prompt = META_EDITOR_TASK.format(
        items_text=items_text,
        content_review=content_review,
        linguistic_review=linguistic_review,
        bias_review=bias_review,
    ) + (
        "\n\nReturn ONLY JSON matching this shape:\n"
        '{"items":[{"item_number":1,"decision":"KEEP","reason":"...","revised_item_stem":null}],'
        '"overall_synthesis":"..."}'
    )

    messages = [
        SystemMessage(content=META_EDITOR_SYSTEM),
        HumanMessage(content=prompt),
    ]

    parsed = await invoke_structured_with_fix(
        agent_name="meta_editor",
        messages=messages,
        schema=MetaEditorOutput,
    )
    review_text = json.dumps(parsed.model_dump(), ensure_ascii=True, indent=2)

    logger.info("meta_editor_done")

    print_agent_message("MetaEditor", "Critic", format_structured_agent_output("MetaEditor", parsed))

    return {"meta_review": review_text}
