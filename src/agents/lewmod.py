"""LewMod Agent: automated LLM-based feedback replacing human-in-the-loop.

Persona: Senior psychometrician with expertise in construct validity,
scale development, and item quality assessment.
Uses temperature=0.3 for balanced, expert-like evaluation.

Activated via `python run.py --lewmod`.
"""

from __future__ import annotations

import sqlite3

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.models import create_llm
from src.persistence.repository import get_latest_round_id, save_feedback
from src.prompts.templates import LEWMOD_SYSTEM, LEWMOD_TASK
from src.schemas.state import MainState
from src.utils.console import print_agent_message, validate_llm_response

logger = structlog.get_logger(__name__)


async def lewmod_node(state: MainState) -> dict:
    """LewMod agent node: provides automated expert feedback or approval.

    Reads items_text and review_text from state, evaluates holistically,
    and either approves (current_phase='done') or requests revision
    (current_phase='revision') with specific feedback.
    """
    items_text = state.get("items_text", "")
    review_text = state.get("review_text", "")
    revision_count = state.get("revision_count", 0)

    logger.info("lewmod_start", revision_count=revision_count)

    llm = create_llm("lewmod")

    prompt = LEWMOD_TASK.format(
        items_text=items_text,
        review_text=review_text,
        revision_count=revision_count,
    )

    messages = [
        SystemMessage(content=LEWMOD_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    feedback_text = validate_llm_response(response.content, "LewMod")

    logger.info("lewmod_done", revision_count=revision_count)

    print_agent_message("LewMod", "Critic", feedback_text)

    # Parse the decision from the response
    # LewMod is instructed to start with "DECISION: APPROVE" or "DECISION: REVISE"
    lower_text = feedback_text.strip().lower()
    is_approved = "decision: approve" in lower_text
    decision = "approve" if is_approved else "revise"

    # Persist feedback to DB
    db_path = state.get("db_path")
    run_id = state.get("run_id")
    if db_path and run_id:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            round_id = get_latest_round_id(conn, run_id)
            if round_id is not None:
                save_feedback(conn, round_id, source="lewmod", feedback_text=feedback_text, decision=decision)
            conn.close()
        except Exception:
            logger.warning("lewmod_db_write_failed", exc_info=True)

    if is_approved:
        return {
            "human_feedback": feedback_text,
            "current_phase": "done",
            "messages": [f"[LewMod] Approved items after {revision_count} revision(s)"],
        }

    return {
        "human_feedback": feedback_text,
        "current_phase": "revision",
        "revision_count": revision_count + 1,
        "messages": [f"[LewMod] Revision requested (round {revision_count + 1})"],
    }
