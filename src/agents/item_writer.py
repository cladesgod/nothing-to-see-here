"""Item Writer Agent: generates and revises Likert-scale test items.

Uses temperature=1.0 for initial generation (creative diversity, per paper).
Uses temperature=0.7 for revision (follow reviewer instructions more closely).
Outputs natural language text (paper-like communication style).
"""

from __future__ import annotations

import re
from contextlib import closing

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.json import parse_json_markdown

from src.config import get_agent_settings
from src.persistence.db import get_connection
from src.persistence.repository import get_previous_items, save_generation_round
from src.prompts.templates import (
    ITEM_WRITER_GENERATE,
    ITEM_WRITER_REVISE,
    ITEM_WRITER_SYSTEM,
)
from src.schemas.agent_outputs import ItemWriterOutput, MetaEditorOutput
from src.schemas.phases import Phase
from src.schemas.state import MainState
from src.utils.console import print_agent_message
from src.utils.structured_output import invoke_structured_with_fix

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


def _parse_human_directives(human_feedback: str) -> tuple[list[int], list[int]]:
    """Parse human directives like 'KEEP: 1,3' and 'REVISE: 2,4'.

    Returns:
        (keep_numbers, revise_numbers)
    """
    if not human_feedback or not human_feedback.strip():
        return [], []

    keep_set: set[int] = set()
    revise_set: set[int] = set()

    for raw in human_feedback.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()

        if upper.startswith("KEEP:"):
            nums = [int(n) for n in re.findall(r"\d+", line)]
            keep_set.update(nums)
        elif upper.startswith("REVISE:"):
            nums = [int(n) for n in re.findall(r"\d+", line)]
            revise_set.update(nums)

    # Human REVISE overrides human KEEP when both provided for same item.
    keep_set -= revise_set
    return sorted(keep_set), sorted(revise_set)


def _get_human_decisions(state: MainState, valid_item_numbers: set[int]) -> tuple[list[int], list[int]]:
    """Read human KEEP/REVISE decisions from structured state with safe fallback."""
    structured = state.get("human_item_decisions", {})
    keep_set: set[int] = set()
    revise_set: set[int] = set()

    if isinstance(structured, dict):
        for key, raw_decision in structured.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if idx not in valid_item_numbers:
                continue
            decision = str(raw_decision).upper().strip()
            if decision == "KEEP":
                keep_set.add(idx)
            elif decision == "REVISE":
                revise_set.add(idx)

    if keep_set or revise_set:
        keep_set -= revise_set  # REVISE overrides KEEP
        return sorted(keep_set), sorted(revise_set)

    # Backward compatibility with any legacy free-text feedback.
    fallback_feedback = state.get("human_feedback", "")
    return _parse_human_directives(fallback_feedback)


def _format_human_feedback_for_prompt(
    keep_numbers: list[int],
    revise_numbers: list[int],
    global_note: str,
) -> str:
    """Create compact human guidance text for revision prompt."""
    lines: list[str] = []
    if keep_numbers:
        lines.append(f"KEEP: {','.join(str(n) for n in keep_numbers)}")
    if revise_numbers:
        lines.append(f"REVISE: {','.join(str(n) for n in revise_numbers)}")
    if global_note:
        lines.append(f"NOTE: {global_note}")
    return "\n".join(lines).strip()


def _parse_numbered_blocks(items_text: str) -> dict[int, str]:
    """Parse numbered item blocks like '1. ...' or '1) ...'."""
    blocks: dict[int, list[str]] = {}
    current_num: int | None = None
    pattern = re.compile(r"^\s*(\d+)[\.\)]\s+(.*)$")

    for raw_line in items_text.splitlines():
        # Keep item parsing isolated from trailing metadata like response scale.
        if raw_line.strip().lower().startswith("response scale:"):
            break
        m = pattern.match(raw_line)
        if m:
            current_num = int(m.group(1))
            blocks[current_num] = [m.group(2).rstrip()]
            continue
        if current_num is not None:
            blocks[current_num].append(raw_line.rstrip())

    parsed: dict[int, str] = {}
    for k, lines in blocks.items():
        text = "\n".join(lines).strip()
        if text:
            parsed[k] = text
    return parsed


def _render_numbered_blocks(blocks: dict[int, str]) -> str:
    """Render parsed numbered blocks back into plain numbered text."""
    lines: list[str] = []
    for idx in sorted(blocks.keys()):
        block_lines = [line for line in blocks[idx].splitlines() if line.strip() != ""]
        if not block_lines:
            continue
        lines.append(f"{idx}. {block_lines[0].strip()}")
        lines.extend(block_lines[1:])
    return "\n".join(lines).strip()


def _extract_response_scale(items_text: str) -> str:
    """Extract trailing response scale line if present."""
    for raw_line in items_text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("response scale:"):
            return line
    return ""


def _subset_blocks(blocks: dict[int, str], numbers: list[int]) -> dict[int, str]:
    """Pick a subset of numbered blocks by item number."""
    return {n: blocks[n] for n in numbers if n in blocks}


def _align_generated_to_targets(
    generated_blocks: dict[int, str],
    target_numbers: list[int],
) -> dict[int, str]:
    """Align generated revision output to target item numbers.

    If model keeps original numbering, use direct matches.
    If model renumbers from 1..N, map by sorted order.
    """
    if not generated_blocks or not target_numbers:
        return {}

    target_set = set(target_numbers)
    if set(generated_blocks.keys()) == target_set:
        return generated_blocks

    generated_key_set = set(generated_blocks.keys())
    # Prefer exact mapping only when all target IDs exist.
    if target_set.issubset(generated_key_set):
        return {n: generated_blocks[n] for n in target_numbers}

    # Fallback: map targets by sorted output order.
    aligned: dict[int, str] = {}
    generated_values = [generated_blocks[k] for k in sorted(generated_blocks.keys())]
    for idx, target_num in enumerate(target_numbers):
        if idx < len(generated_values):
            aligned[target_num] = generated_values[idx]
    return aligned


async def _extract_keep_numbers(review_text: str) -> list[int]:
    """Extract KEEP item numbers from meta editor structured JSON."""
    try:
        direct = MetaEditorOutput.model_validate(parse_json_markdown(review_text))
        return sorted({d.item_number for d in direct.items if d.decision == "KEEP"})
    except Exception:
        pass

    messages = [
        SystemMessage(
            content=(
                "Validate and normalize this meta editor output to schema. "
                "Return ONLY valid JSON."
            )
        ),
        HumanMessage(content=review_text),
    ]
    try:
        bundle = await invoke_structured_with_fix(
            agent_name="meta_editor",
            messages=messages,
            schema=MetaEditorOutput,
            max_attempts=4,
        )
    except Exception:
        return []

    keep_numbers = sorted({d.item_number for d in bundle.items if d.decision == "KEEP"})
    return keep_numbers


def _format_locked_items(original_items_text: str, keep_numbers: list[int]) -> str:
    """Build explicit lock instructions for KEEP items."""
    if not keep_numbers:
        return ""
    original_blocks = _parse_numbered_blocks(original_items_text)
    locked_lines = ["**Locked KEEP Items (must remain EXACTLY unchanged):**"]
    for num in keep_numbers:
        text = original_blocks.get(num)
        if text:
            first_line = text.splitlines()[0]
            locked_lines.append(f"- Item {num}: {first_line}")
    if len(locked_lines) == 1:
        return ""
    return "\n".join(locked_lines) + "\n"


def _enforce_keep_locks(
    original_items_text: str,
    generated_items_text: str,
    keep_numbers: list[int],
) -> str:
    """Force KEEP items to stay unchanged after generation."""
    if not keep_numbers:
        return generated_items_text

    original_blocks = _parse_numbered_blocks(original_items_text)
    generated_blocks = _parse_numbered_blocks(generated_items_text)
    if not original_blocks or not generated_blocks:
        return generated_items_text

    changed = False
    for num in keep_numbers:
        orig = original_blocks.get(num)
        if not orig:
            continue
        if generated_blocks.get(num) != orig:
            generated_blocks[num] = orig
            changed = True

    if not changed:
        return generated_items_text
    return _render_numbered_blocks(generated_blocks)


async def item_writer_node(state: MainState) -> dict:
    """Item Writer agent node: generates or revises items based on phase."""
    construct_name = state.get("construct_name", "")
    construct_definition = state.get("construct_definition", "")
    research_summary = state.get("research_summary", "")
    current_phase = state.get("current_phase", Phase.ITEM_GENERATION)
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
            with closing(get_connection(db_path)) as conn:
                construct_fingerprint = state.get("construct_fingerprint", "")
                previous_from_db = get_previous_items(
                    conn,
                    construct_fingerprint,
                    exclude_run_id=run_id,
                    limit=workflow_cfg.memory_limit,
                )
        except Exception:
            logger.warning("item_writer_db_read_failed", exc_info=True)

    all_previous = previous_from_db + state.get("previously_approved_items", [])
    history_text = _format_item_history(all_previous)

    previous_round_items = ""
    frozen_item_numbers = sorted(set(state.get("frozen_item_numbers", [])))
    active_items_text = state.get("items_text", "")
    if current_phase == Phase.REVISION:
        # Revision mode: use reviewer feedback to improve items
        items_text = state.get("items_text", "")
        previous_round_items = items_text.strip()
        review_text = state.get("review_text", "")
        meta_keep = await _extract_keep_numbers(review_text)
        all_blocks = _parse_numbered_blocks(items_text)
        human_keep, human_revise = _get_human_decisions(state, set(all_blocks.keys()))
        global_note = state.get("human_global_note", "")
        human_feedback = _format_human_feedback_for_prompt(human_keep, human_revise, global_note)
        new_keep = sorted((set(meta_keep) | set(human_keep)) - set(human_revise))
        frozen_item_numbers = sorted((set(frozen_item_numbers) | set(new_keep)) - set(human_revise))

        original_blocks = all_blocks
        all_numbers = sorted(original_blocks.keys())
        active_numbers = [n for n in all_numbers if n not in set(frozen_item_numbers)]
        active_blocks = _subset_blocks(original_blocks, active_numbers)
        active_items_text = _render_numbered_blocks(active_blocks)

        locked_items = _format_locked_items(items_text, frozen_item_numbers)

        if not active_numbers:
            logger.info("item_writer_all_items_frozen_skip_revision")
            print_agent_message(
                "ItemWriter",
                "Critic",
                "All items are frozen as KEEP. Skipping revision generation.",
            )
            return {
                "items_text": items_text,
                "active_items_text": "",
                "frozen_item_numbers": frozen_item_numbers,
                "current_phase": Phase.REVIEW,
                "messages": ["[ItemWriter] Skipped revision because all items are frozen"],
                "previously_approved_items": [previous_round_items] if previous_round_items else [],
            }

        prompt = ITEM_WRITER_REVISE.format(
            construct_name=construct_name,
            construct_definition=construct_definition,
            items_text=active_items_text,
            review_text=review_text,
            human_feedback=human_feedback or "No human feedback provided.",
        )
        if locked_items:
            prompt += f"\n{locked_items}\n"
        if history_text:
            prompt += f"\n{history_text}\n"
        prompt += (
            "\n\nReturn ONLY JSON with schema:\n"
            '{"items":[{"item_number":1,"stem":"...","rationale":"..."}],'
            '"response_scale":"1 (Strongly Disagree) to 7 (Strongly Agree)"}'
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
        ) + (
            "\n\nReturn ONLY JSON with schema:\n"
            '{"items":[{"item_number":1,"stem":"...","rationale":"..."}],'
            '"response_scale":"1 (Strongly Disagree) to 7 (Strongly Agree)"}'
        )

    messages = [
        SystemMessage(content=ITEM_WRITER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    parsed = await invoke_structured_with_fix(
        agent_name="item_writer",
        messages=messages,
        schema=ItemWriterOutput,
    )
    generated_items_text = "\n".join(
        f"{item.item_number}. {item.stem} Rationale: {item.rationale}" for item in parsed.items
    ).strip()
    if parsed.response_scale:
        generated_items_text += f"\n\nResponse scale: {parsed.response_scale}"

    items_text = generated_items_text
    active_items_text = generated_items_text
    if current_phase == Phase.REVISION:
        original_blocks = _parse_numbered_blocks(previous_round_items)
        active_numbers = [n for n in sorted(original_blocks.keys()) if n not in set(frozen_item_numbers)]
        generated_blocks = _parse_numbered_blocks(generated_items_text)
        aligned_revised = _align_generated_to_targets(generated_blocks, active_numbers)
        merged_blocks = dict(original_blocks)
        merged_blocks.update(aligned_revised)
        items_text = _render_numbered_blocks(merged_blocks)
        active_items_text = _render_numbered_blocks(_subset_blocks(merged_blocks, active_numbers))
        response_scale = parsed.response_scale or _extract_response_scale(previous_round_items)
        if response_scale:
            items_text = f"{items_text}\n\nResponse scale: {response_scale}"
            if active_items_text:
                active_items_text = f"{active_items_text}\n\nResponse scale: {response_scale}"
    elif parsed.response_scale:
        items_text = generated_items_text
        active_items_text = generated_items_text

    logger.info("item_writer_done", phase=current_phase)

    print_agent_message("ItemWriter", "Critic", items_text)

    # Persist generation round to DB
    if db_path and run_id:
        try:
            with closing(get_connection(db_path)) as conn:
                save_generation_round(
                    conn,
                    run_id=run_id,
                    round_number=state.get("revision_count", 0),
                    phase=current_phase,
                    items_text=items_text,
                )
        except Exception:
            logger.warning("item_writer_db_write_failed", exc_info=True)

    output = {
        "items_text": items_text,
        "active_items_text": active_items_text,
        "frozen_item_numbers": frozen_item_numbers,
        "current_phase": Phase.REVIEW,
        "messages": [f"[ItemWriter] Generated items ({current_phase} mode)"],
    }
    if previous_round_items:
        output["previously_approved_items"] = [previous_round_items]
    return output
