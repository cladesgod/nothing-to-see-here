"""Main Workflow Graph: orchestrates the full item generation pipeline.

Matches the paper's architecture (Fig. 1 & Fig. 2):

                    ┌──→ web_surfer ──┐
                    │                 │
  START → critic ──┼──→ item_writer ──┼──→ critic ──→ END (output)
            ↑       │                 │       ↑
            │       ├──→ review_chain ─┤       │
            │       │   (subgraph)    │       │
            │       └──→ human_feedback┘       │
            │                                  │
            └──────────────────────────────────┘

The Critic Agent is the central hub / supervisor.
Every worker node feeds back into the Critic, which decides the next step.

All agent communication uses natural language text (paper-like style).
"""

from __future__ import annotations

import json
from contextlib import closing

import structlog
from langchain_core.utils.json import parse_json_markdown
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy, interrupt

from src.agents.critic import critic_node, critic_router
from src.agents.item_writer import item_writer_node
from src.agents.lewmod import lewmod_node
from src.agents.web_surfer import web_surfer_node
from src.config import get_agent_settings
from src.graphs.review_chain import review_chain_graph
from src.persistence.db import get_connection
from src.persistence.repository import get_latest_round_id, save_feedback, save_review
from src.schemas.agent_outputs import MetaEditorOutput
from src.schemas.phases import Phase
from src.schemas.state import MainState, ReviewChainState
from src.utils.console import (
    format_structured_agent_output,
    print_agent_message,
    print_human_prompt,
)
from src.utils.deterministic_scoring import build_deterministic_meta_review

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Review chain wrapper: runs the subgraph for all items as a batch
# ---------------------------------------------------------------------------

async def review_chain_wrapper(state: MainState) -> dict:
    """Run the review chain subgraph for the batch of items."""
    items_text = state.get("items_text", "")
    active_items_text = state.get("active_items_text", items_text)
    construct_name = state.get("construct_name", "")
    construct_definition = state.get("construct_definition", "")
    dimension_info = state.get("dimension_info", "")

    logger.info("review_chain_start")

    if not active_items_text.strip():
        logger.info("review_chain_skip_no_active_items")
        review_text = (
            '{\n'
            '  "items": [],\n'
            '  "overall_synthesis": "All items are frozen as KEEP. No active items to review in this round."\n'
            '}'
        )
        return {
            "review_text": review_text,
            "current_phase": Phase.HUMAN_FEEDBACK,
            "messages": ["[ReviewChain] Skipped (no active items)"],
        }

    # Run the review chain subgraph with active items only
    sub_state: ReviewChainState = {
        "items_text": active_items_text,
        "construct_name": construct_name,
        "construct_definition": construct_definition,
        "dimension_info": dimension_info,
    }

    result = await review_chain_graph.ainvoke(sub_state)
    review_text = result.get("meta_review", "")
    if not review_text or not review_text.strip():
        logger.warning("review_chain_empty_output")
        review_text = (
            "WARNING: The review chain did not produce a substantive review. "
            "This may indicate a failure in one or more reviewer agents. "
            "The items should be manually inspected before proceeding."
        )
    else:
        deterministic_meta = build_deterministic_meta_review(
            content_review_text=result.get("content_review", ""),
            linguistic_review_text=result.get("linguistic_review", ""),
            bias_review_text=result.get("bias_review", ""),
            meta_review_text=review_text,
        )
        review_text = deterministic_meta.model_dump_json(indent=2)

    logger.info("review_chain_done")

    # Persist review results to DB
    db_path = state.get("db_path")
    run_id = state.get("run_id")
    if db_path and run_id:
        try:
            with closing(get_connection(db_path)) as conn:
                round_id = get_latest_round_id(conn, run_id)
                if round_id is not None:
                    # Keep raw reviewer/meta payloads for auditability.
                    # Downstream state uses deterministic_meta JSON.
                    save_review(
                        conn,
                        round_id,
                        content_review=result.get("content_review", ""),
                        linguistic_review=result.get("linguistic_review", ""),
                        bias_review=result.get("bias_review", ""),
                        meta_review=result.get("meta_review", ""),
                    )
        except Exception:
            logger.warning("review_chain_db_write_failed", exc_info=True)

    return {
        "review_text": review_text,
        "current_phase": Phase.HUMAN_FEEDBACK,
        "messages": ["[ReviewChain] Review completed"],
    }


# ---------------------------------------------------------------------------
# Human feedback node: uses interrupt() for human-in-the-loop
# ---------------------------------------------------------------------------

async def human_feedback_node(state: MainState) -> dict:
    """Collect human feedback via interrupt.

    Pauses the graph execution and waits for human input.
    Resume with: graph.invoke(Command(resume="your feedback here"), config)
    """
    items_text = state.get("items_text", "")
    active_items_text = state.get("active_items_text", "") or items_text
    frozen_numbers = sorted(set(state.get("frozen_item_numbers", [])))
    review_text = state.get("review_text", "")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)
    review_display = review_text
    try:
        parsed = MetaEditorOutput.model_validate(parse_json_markdown(review_text))
        review_display = format_structured_agent_output("MetaEditor", parsed)
    except Exception:
        pass

    # Build a summary for the human reviewer
    frozen_note = ""
    if frozen_numbers:
        frozen_note = (
            f"**Frozen KEEP items (auto-kept):** "
            f"{', '.join(str(n) for n in frozen_numbers)}\n\n"
        )

    summary = (
        f"## Active Items for Review\n\n{active_items_text}\n\n"
        f"---\n\n"
        f"## Meta Editor Review\n\n{review_display}\n\n"
        f"---\n\n"
        f"{frozen_note}"
        f"**Revision round:** {revision_count + 1}/{max_revisions}\n\n"
        "Use interactive CLI controls to set KEEP/REVISE per item.\n"
        "Frozen KEEP items are not asked again.\n"
        "You can also provide one global note.\n"
    )

    print_human_prompt(summary)

    # Interrupt: pause and wait for human input
    human_input = interrupt(summary)

    # Process the human response
    structured_decisions: dict[str, str] = {}
    global_note = ""
    if isinstance(human_input, dict):
        is_approved = bool(human_input.get("approve", False))
        raw_decisions = human_input.get("item_decisions", {})
        if isinstance(raw_decisions, dict):
            for key, value in raw_decisions.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                decision = str(value).upper().strip()
                if decision in {"KEEP", "REVISE"}:
                    structured_decisions[str(idx)] = decision
        raw_note = human_input.get("global_note", "")
        global_note = str(raw_note).strip() if raw_note else ""
        feedback = json.dumps(
            {
                "approve": is_approved,
                "item_decisions": dict(sorted(structured_decisions.items())),
                "global_note": global_note,
            },
            ensure_ascii=True,
        )
    else:
        feedback = human_input if isinstance(human_input, str) else str(human_input)
        is_approved = feedback.strip().lower() == "approve"
    decision = "approve" if is_approved else "revise"

    # === INJECTION DEFENSE ===
    if global_note and not is_approved:
        from src.utils.injection_defense import check_prompt_injection

        is_safe, rejection_msg = await check_prompt_injection(global_note)
        if not is_safe:
            logger.warning(
                "injection_defense_blocked", input_length=len(global_note)
            )
            print_agent_message("System", "Human", rejection_msg)
            return {
                "human_feedback": "blocked",
                "human_item_decisions": {},
                "human_global_note": "",
                "current_phase": Phase.DONE,
                "messages": [
                    "[System] Run terminated: prompt injection detected in feedback"
                ],
            }

    print_agent_message("Human", "Critic", "Approved all items." if is_approved else feedback)

    # Persist feedback to DB
    db_path = state.get("db_path")
    run_id = state.get("run_id")
    if db_path and run_id:
        try:
            with closing(get_connection(db_path)) as conn:
                round_id = get_latest_round_id(conn, run_id)
                if round_id is not None:
                    save_feedback(conn, round_id, source="human", feedback_text=feedback, decision=decision)
        except Exception:
            logger.warning("human_feedback_db_write_failed", exc_info=True)

    if is_approved:
        return {
            "human_feedback": "approved",
            "human_item_decisions": {},
            "human_global_note": "",
            "current_phase": Phase.DONE,
            "messages": ["[Human] Approved all items"],
        }

    return {
        "human_feedback": feedback,
        "human_item_decisions": structured_decisions,
        "human_global_note": global_note,
        "current_phase": Phase.REVISION,
        "revision_count": revision_count + 1,
        "messages": [f"[Human] Provided feedback for revision round {revision_count + 1}"],
    }


# ---------------------------------------------------------------------------
# Build the main workflow graph
# ---------------------------------------------------------------------------

_CRITIC_EDGES = {
    "web_surfer": "web_surfer",
    "item_writer": "item_writer",
    "review_chain": "review_chain",
    "human_feedback": "human_feedback",
    "done": END,
}


def build_main_workflow(checkpointer=None, lewmod=False, db_url: str | None = None):
    """Build and compile the main workflow graph.

    Architecture (matching paper Fig. 1 & 2):
      - Critic Agent is a visible central node
      - All worker nodes → Critic → conditional routing
      - Review chain is a subgraph with parallel reviewers

    Args:
        checkpointer: LangGraph checkpointer for persistence.
                      - None  → use MemorySaver (default, for CLI / standalone)
                      - False → no checkpointer (for LangGraph Platform)
                      - "postgres" → use PostgresSaver (requires db_url)
                      - Any Checkpointer instance → use that directly
        lewmod: If True, use LewMod (automated LLM feedback) instead of
                human-in-the-loop feedback. Default: False.
        db_url: Database URL for PostgreSQL checkpointer. Only used when
                checkpointer="postgres".
    """
    builder = StateGraph(MainState)

    # ---- Retry policy for LLM nodes (configurable via agents.toml) ----
    agent_settings = get_agent_settings()
    retry = RetryPolicy(
        max_attempts=agent_settings.retry.max_attempts,
        initial_interval=agent_settings.retry.initial_interval,
        backoff_factor=agent_settings.retry.backoff_factor,
    )

    # ---- Nodes ----
    builder.add_node("critic", critic_node)  # Deterministic — no retry
    builder.add_node("web_surfer", web_surfer_node, retry_policy=retry)
    builder.add_node("item_writer", item_writer_node, retry_policy=retry)
    builder.add_node("review_chain", review_chain_wrapper, retry_policy=retry)
    if lewmod:
        builder.add_node("human_feedback", lewmod_node, retry_policy=retry)
    else:
        builder.add_node("human_feedback", human_feedback_node)  # Interrupt — no retry

    # ---- Edges ----
    # START → Critic (always enters through critic first)
    builder.add_edge(START, "critic")

    # Critic → conditional routing to workers or END
    builder.add_conditional_edges("critic", critic_router, _CRITIC_EDGES)

    # All workers → back to Critic (the central hub pattern from Fig. 1)
    builder.add_edge("web_surfer", "critic")
    builder.add_edge("item_writer", "critic")
    builder.add_edge("review_chain", "critic")
    builder.add_edge("human_feedback", "critic")

    # ---- Resolve checkpointer ----
    if checkpointer == "postgres" and db_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            pg_checkpointer = PostgresSaver.from_conn_string(db_url)
            pg_checkpointer.setup()
            checkpointer = pg_checkpointer
            logger.info("postgres_checkpointer_configured")
        except ImportError:
            logger.warning("postgres_checkpointer_unavailable_falling_back_to_memory")
            checkpointer = MemorySaver()
    elif checkpointer is None:
        checkpointer = MemorySaver()
    elif checkpointer is False:
        checkpointer = None  # LangGraph Platform provides its own

    return builder.compile(checkpointer=checkpointer)


# Module-level graph instance for `langgraph dev` / LangGraph Studio.
# Referenced in langgraph.json as "aig_workflow": "./src/graphs/main_workflow.py:graph"
# NOTE: No checkpointer here — LangGraph Platform provides its own persistence.
graph = build_main_workflow(checkpointer=False)
