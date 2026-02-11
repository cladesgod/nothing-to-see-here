"""LewMod Agent: automated LLM-based feedback replacing human-in-the-loop.

Persona: Senior psychometrician with expertise in construct validity,
scale development, and item quality assessment.
Uses temperature=0.3 for balanced, expert-like evaluation.

Activated via `python run.py --lewmod`.
"""

from __future__ import annotations

import re
from contextlib import closing

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.persistence.db import get_connection
from src.persistence.repository import get_latest_round_id, save_feedback
from src.prompts.templates import LEWMOD_SYSTEM, LEWMOD_TASK
from src.schemas.agent_outputs import LewModOutput
from src.schemas.phases import Phase
from src.schemas.state import MainState
from src.utils.console import print_agent_message
from src.utils.structured_output import invoke_structured_with_fix

logger = structlog.get_logger(__name__)


def _parse_numbered_item_ids(items_text: str) -> set[int]:
    """Extract item numbers from numbered plain-text blocks."""
    ids: set[int] = set()
    for line in items_text.splitlines():
        m = re.match(r"^\s*(\d+)[\.\)]\s+", line)
        if m:
            ids.add(int(m.group(1)))
    return ids


def _build_item_decisions_map(
    parsed: LewModOutput,
    allowed_item_ids: set[int] | None = None,
) -> dict[str, str]:
    """Normalize LewMod keep/revise/discard lists to state decision map."""
    decisions: dict[str, str] = {}

    def _allowed(num: int) -> bool:
        return allowed_item_ids is None or num in allowed_item_ids

    for num in parsed.keep:
        if _allowed(num):
            decisions[str(num)] = "KEEP"
    for num in parsed.revise:
        if _allowed(num):
            decisions[str(num)] = "REVISE"
    # DISCARD currently routes to revision/replacement in item writer.
    for num in parsed.discard:
        if _allowed(num):
            decisions[str(num)] = "REVISE"
    return decisions


async def lewmod_node(state: MainState) -> dict:
    """LewMod agent node: provides automated expert feedback or approval.

    Reads items_text and review_text from state, evaluates holistically,
    and either approves (current_phase='done') or requests revision
    (current_phase='revision') with specific feedback.
    """
    items_text = state.get("items_text", "")
    active_items_text = (
        state["active_items_text"] if "active_items_text" in state else items_text
    )
    review_text = state.get("review_text", "")
    revision_count = state.get("revision_count", 0)
    active_item_ids = _parse_numbered_item_ids(active_items_text)

    logger.info("lewmod_start", revision_count=revision_count)

    if not active_items_text.strip():
        logger.info("lewmod_no_active_items_auto_approve")
        print_agent_message("LewMod", "Critic", "No active items left. Auto-approving.")
        return {
            "human_feedback": "No active items left. Auto-approved.",
            "human_item_decisions": {},
            "human_global_note": "",
            "current_phase": Phase.DONE,
            "messages": [f"[LewMod] Approved items after {revision_count} revision(s)"],
        }

    prompt = LEWMOD_TASK.format(
        items_text=active_items_text,
        review_text=review_text,
        revision_count=revision_count,
    ) + (
        "\n\nReturn ONLY JSON with schema:\n"
        '{"decision":"APPROVE|REVISE","feedback":"...","keep":[1],"revise":[2],"discard":[3]}'
    )

    messages = [
        SystemMessage(content=LEWMOD_SYSTEM),
        HumanMessage(content=prompt),
    ]

    parsed = await invoke_structured_with_fix(
        agent_name="lewmod",
        messages=messages,
        schema=LewModOutput,
    )
    feedback_text = parsed.feedback.strip()

    logger.info("lewmod_done", revision_count=revision_count)

    print_agent_message("LewMod", "Critic", feedback_text)

    is_approved = parsed.decision == "APPROVE"
    decision = "approve" if is_approved else "revise"
    item_decisions = _build_item_decisions_map(
        parsed,
        allowed_item_ids=active_item_ids if active_item_ids else None,
    )

    # Persist feedback to DB
    db_path = state.get("db_path")
    run_id = state.get("run_id")
    if db_path and run_id:
        try:
            with closing(get_connection(db_path)) as conn:
                round_id = get_latest_round_id(conn, run_id)
                if round_id is not None:
                    save_feedback(conn, round_id, source="lewmod", feedback_text=feedback_text, decision=decision)
        except Exception:
            logger.warning("lewmod_db_write_failed", exc_info=True)

    if is_approved:
        return {
            "human_feedback": feedback_text,
            "human_item_decisions": {},
            "human_global_note": "",
            "current_phase": Phase.DONE,
            "messages": [f"[LewMod] Approved items after {revision_count} revision(s)"],
        }

    return {
        "human_feedback": feedback_text,
        "human_item_decisions": item_decisions,
        "human_global_note": "",
        "current_phase": Phase.REVISION,
        "revision_count": revision_count + 1,
        "messages": [f"[LewMod] Revision requested (round {revision_count + 1})"],
    }
