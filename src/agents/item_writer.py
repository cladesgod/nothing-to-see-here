"""Item Writer Agent: generates and revises Likert-scale test items.

Uses temperature=1.0 for creative diversity (per paper guidelines).
Supports two modes: initial generation and revision based on feedback.
Outputs natural language text (paper-like communication style).
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_agent_settings
from src.models import create_llm
from src.prompts.templates import (
    ITEM_WRITER_GENERATE,
    ITEM_WRITER_REVISE,
    ITEM_WRITER_SYSTEM,
)
from src.schemas.state import MainState
from src.utils.console import print_agent_message

logger = structlog.get_logger(__name__)


async def item_writer_node(state: MainState) -> dict:
    """Item Writer agent node: generates or revises items based on phase."""
    construct_name = state.get("construct_name", "")
    construct_definition = state.get("construct_definition", "")
    research_summary = state.get("research_summary", "")
    current_phase = state.get("current_phase", "item_generation")

    logger.info(
        "item_writer_start",
        phase=current_phase,
        construct=construct_name,
    )

    llm = create_llm("item_writer")

    if current_phase == "revision":
        # Revision mode: use reviewer feedback to improve items
        items_text = state.get("items_text", "")
        review_text = state.get("review_text", "")
        human_feedback = state.get("human_feedback", "")

        prompt = ITEM_WRITER_REVISE.format(
            items_text=items_text,
            review_text=review_text,
            human_feedback=human_feedback or "No human feedback provided.",
        )
    else:
        # Initial generation mode
        agent_cfg = get_agent_settings().get_agent_config("item_writer")
        prompt = ITEM_WRITER_GENERATE.format(
            num_items=agent_cfg.num_items,
            construct_name=construct_name,
            construct_definition=construct_definition,
            dimension_name="(across all dimensions)",
            dimension_definition="See research summary.",
            research_summary=research_summary or "No research available.",
        )

    messages = [
        SystemMessage(content=ITEM_WRITER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    items_text = response.content

    logger.info("item_writer_done", phase=current_phase)

    print_agent_message("ItemWriter", "Critic", items_text)

    return {
        "items_text": items_text,
        "current_phase": "review",
        "messages": [f"[ItemWriter] Generated items ({current_phase} mode)"],
    }
