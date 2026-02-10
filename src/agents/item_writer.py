"""Item Writer Agent: generates and revises Likert-scale test items.

Uses temperature=1.0 for initial generation (creative diversity, per paper).
Uses temperature=0.7 for revision (follow reviewer instructions more closely).
Outputs natural language text (paper-like communication style).
"""

from __future__ import annotations

import sqlite3

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_agent_settings
from src.models import create_llm
from src.persistence.repository import get_previous_items, save_generation_round
from src.prompts.templates import (
    ITEM_WRITER_GENERATE,
    ITEM_WRITER_REVISE,
    ITEM_WRITER_SYSTEM,
)
from src.schemas.state import MainState
from src.utils.console import print_agent_message, validate_llm_response

logger = structlog.get_logger(__name__)


def _format_item_history(previous_items: list[str]) -> str:
    """Format previously generated items for prompt injection."""
    if not previous_items:
        return ""

    header = (
        "**Previously Generated Items (avoid similarity to these):**\n\n"
        "If prior items are listed below, ensure your new items are conceptually distinct.\n"
        "Avoid generating items that:\n"
        "- Overlap in meaning or phrasing with prior items\n"
        "- Tap into the same narrow behavioral facet as a prior item\n"
        "- Could be confused with a prior item by a naive rater\n\n"
    )
    sections = []
    for i, items_text in enumerate(previous_items, 1):
        sections.append(f"--- Prior Set {i} ---\n{items_text}")

    return header + "\n\n".join(sections)


async def item_writer_node(state: MainState) -> dict:
    """Item Writer agent node: generates or revises items based on phase."""
    construct_name = state.get("construct_name", "")
    construct_definition = state.get("construct_definition", "")
    research_summary = state.get("research_summary", "")
    current_phase = state.get("current_phase", "item_generation")
    db_path = state.get("db_path")
    run_id = state.get("run_id")

    logger.info(
        "item_writer_start",
        phase=current_phase,
        construct=construct_name,
    )

    # Gather previously generated items for anti-homogeneity
    workflow_cfg = get_agent_settings().workflow
    previous_from_db: list[str] = []
    if workflow_cfg.memory_enabled and db_path:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            previous_from_db = get_previous_items(
                conn, construct_name, exclude_run_id=run_id, limit=workflow_cfg.memory_limit,
            )
            conn.close()
        except Exception:
            logger.warning("item_writer_db_read_failed", exc_info=True)

    all_previous = previous_from_db + state.get("previously_approved_items", [])
    history_text = _format_item_history(all_previous)

    # Lower temperature in revision mode so the model follows reviewer
    # instructions more faithfully instead of regenerating from scratch.
    revision_temp = 0.7 if current_phase == "revision" else None
    llm = create_llm("item_writer", temperature=revision_temp)

    if current_phase == "revision":
        # Revision mode: use reviewer feedback to improve items
        items_text = state.get("items_text", "")
        review_text = state.get("review_text", "")
        human_feedback = state.get("human_feedback", "")

        prompt = ITEM_WRITER_REVISE.format(
            items_text=items_text,
            review_text=review_text,
            human_feedback=human_feedback or "No human feedback provided.",
            previously_approved_items=history_text,
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
            previously_approved_items=history_text,
        )

    messages = [
        SystemMessage(content=ITEM_WRITER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    items_text = validate_llm_response(response.content, "ItemWriter")

    logger.info("item_writer_done", phase=current_phase)

    print_agent_message("ItemWriter", "Critic", items_text)

    # Persist generation round to DB
    if db_path and run_id:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            save_generation_round(
                conn,
                run_id=run_id,
                round_number=state.get("revision_count", 0),
                phase=current_phase,
                items_text=items_text,
            )
            conn.close()
        except Exception:
            logger.warning("item_writer_db_write_failed", exc_info=True)

    return {
        "items_text": items_text,
        "current_phase": "review",
        "messages": [f"[ItemWriter] Generated items ({current_phase} mode)"],
    }
