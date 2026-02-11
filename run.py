"""CLI entry point for the LM-AIG Multi-Agent System.

Usage:
    python run.py                                  # Run with AAAW preset (default)
    python run.py --lewmod                         # Run with LewMod (automated feedback)
    python run.py --preset aaaw                    # Explicit preset selection
    python run.py --construct-file my_construct.json  # Custom construct from JSON
    python run.py --json my_construct.json         # Short alias for construct JSON
    python run.py --json                           # Uses examples/custom_construct.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import uuid

from langchain_core.utils.json import parse_json_markdown
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from src.config import get_agent_settings, get_settings
from src.graphs.main_workflow import build_main_workflow
from src.logging_config import setup_logging
from src.persistence.db import DB_PATH, get_connection
from src.persistence.repository import create_run, finish_run
from src.schemas.constructs import (
    build_dimension_info,
    compute_fingerprint,
    get_preset,
    list_presets,
    load_construct_from_file,
)
from src.schemas.agent_outputs import MetaEditorOutput
from src.schemas.phases import Phase
from src.utils.deterministic_scoring import build_deterministic_meta_review
from src.utils.console import (
    console,
    format_structured_agent_output,
    print_final_results,
    print_header,
    print_info,
    print_langsmith_status,
    set_verbose_json_output,
)


def _parse_numbered_item_stems(items_text: str) -> dict[int, str]:
    """Parse numbered item stems from plain text."""
    stems: dict[int, str] = {}
    pattern = re.compile(r"^\s*(\d+)[\.\)]\s+(.*)$")
    for line in items_text.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        stems[int(m.group(1))] = m.group(2).strip()
    return stems


def _parse_number_list(raw: str, valid_ids: set[int]) -> list[int]:
    """Parse comma/space separated numbers, filtered to valid IDs."""
    nums = {int(n) for n in re.findall(r"\d+", raw)}
    return sorted(n for n in nums if n in valid_ids)


def _extract_meta_suggestions(review_text: str) -> dict[int, str]:
    """Extract per-item meta decisions from review JSON."""
    if not review_text.strip():
        return {}
    try:
        parsed = MetaEditorOutput.model_validate(parse_json_markdown(review_text))
    except Exception:
        return {}
    return {item.item_number: item.decision for item in parsed.items}


def _extract_metrics_from_reason(reason: str) -> dict[str, str]:
    """Extract deterministic metrics from meta decision reason text."""
    metrics: dict[str, str] = {}
    if not reason:
        return metrics

    c_match = re.search(r"c=([0-9]+(?:\.[0-9]+)?)", reason)
    d_match = re.search(r"d=(-?[0-9]+(?:\.[0-9]+)?)", reason)
    ling_match = re.search(r"ling_min=([0-9]+)", reason)
    bias_match = re.search(r"bias=([0-9]+)", reason)

    if c_match:
        metrics["c"] = c_match.group(1)
    if d_match:
        metrics["d"] = d_match.group(1)
    if ling_match:
        metrics["ling_min"] = ling_match.group(1)
    if bias_match:
        metrics["bias"] = bias_match.group(1)
    return metrics


def _load_keep_metrics_from_run_history(
    conn: sqlite3.Connection,
    run_id: str,
    target_item_ids: set[int],
) -> dict[int, dict[str, str]]:
    """Backfill KEEP metrics from earlier rounds in this run."""
    if not target_item_ids:
        return {}

    rows = conn.execute(
        """
        SELECT
            gr.round_number,
            rv.content_review,
            rv.linguistic_review,
            rv.bias_review,
            rv.meta_review
        FROM reviews rv
        JOIN generation_rounds gr ON rv.round_id = gr.id
        WHERE gr.run_id = ?
        ORDER BY gr.round_number DESC, rv.id DESC
        """,
        (run_id,),
    ).fetchall()

    found: dict[int, dict[str, str]] = {}
    for row in rows:
        deterministic = build_deterministic_meta_review(
            content_review_text=row["content_review"] or "",
            linguistic_review_text=row["linguistic_review"] or "",
            bias_review_text=row["bias_review"] or "",
            meta_review_text=row["meta_review"] or "",
        )
        for decision in deterministic.items:
            if decision.item_number not in target_item_ids:
                continue
            if decision.decision != "KEEP":
                continue
            if decision.item_number in found:
                continue
            found[decision.item_number] = _extract_metrics_from_reason(decision.reason)
        if len(found) == len(target_item_ids):
            break
    return found


def _build_keep_metrics_section(
    items_text: str,
    review_text: str,
    *,
    frozen_item_numbers: list[int] | None = None,
    conn: sqlite3.Connection | None = None,
    run_id: str | None = None,
) -> str:
    """Build final report section listing KEEP items and their metrics."""
    if not items_text.strip():
        return ""

    stems = _parse_numbered_item_stems(items_text)
    keep_ids: set[int] = set(frozen_item_numbers or [])
    metrics_by_id: dict[int, dict[str, str]] = {}

    parsed: MetaEditorOutput | None = None
    if review_text.strip():
        try:
            parsed = MetaEditorOutput.model_validate(parse_json_markdown(review_text))
        except Exception:
            parsed = None

    if parsed is not None:
        for decision in parsed.items:
            if decision.decision == "KEEP":
                keep_ids.add(decision.item_number)
                metrics_by_id[decision.item_number] = _extract_metrics_from_reason(decision.reason)

    if not keep_ids:
        return (
            "## Final KEEP Items & Metrics\n\n"
            "No KEEP items found in final review or frozen-history for this run."
        )

    missing_metric_ids = {
        item_id
        for item_id in keep_ids
        if item_id not in metrics_by_id or not metrics_by_id[item_id]
    }
    if missing_metric_ids and conn is not None and run_id:
        try:
            history_metrics = _load_keep_metrics_from_run_history(conn, run_id, missing_metric_ids)
            for item_id, vals in history_metrics.items():
                if vals:
                    metrics_by_id[item_id] = vals
        except Exception:
            pass

    lines = ["## Final KEEP Items & Metrics", ""]
    for item_id in sorted(keep_ids):
        stem = stems.get(item_id, "(item text unavailable)")
        metrics = metrics_by_id.get(item_id, {})
        metrics_text = (
            f"c={metrics.get('c', '-')}, "
            f"d={metrics.get('d', '-')}, "
            f"ling_min={metrics.get('ling_min', '-')}, "
            f"bias={metrics.get('bias', '-')}"
        )
        lines.append(f"- Item {item_id}: {stem}")
        lines.append(f"  Metrics: {metrics_text}")
    if any(not metrics_by_id.get(item_id) for item_id in keep_ids):
        lines.append("")
        lines.append(
            "Note: '-' indicates deterministic metrics were not found in stored review history for that item."
        )
    return "\n".join(lines)


def _collect_human_feedback(state_values: dict) -> dict[str, object] | str:
    """Collect structured human feedback using item-by-item selection."""
    active_items_text = state_values.get("active_items_text", "")
    items_text = active_items_text or state_values.get("items_text", "")
    stems = _parse_numbered_item_stems(items_text)
    frozen_numbers = sorted(
        {
            int(n)
            for n in state_values.get("frozen_item_numbers", [])
            if isinstance(n, int) or (isinstance(n, str) and n.isdigit())
        }
    )
    if frozen_numbers:
        console.print(
            "[dim]Frozen KEEP items (auto-kept, not shown below): "
            f"{', '.join(str(n) for n in frozen_numbers)}[/dim]"
        )
    if not stems:
        console.print("[green]No active items to revise. Auto-approving this round.[/green]")
        return "approve"

    meta_suggestions = _extract_meta_suggestions(state_values.get("review_text", ""))
    item_decisions: dict[int, str] = {}
    sorted_ids = sorted(stems.keys())
    valid_ids = set(sorted_ids)

    console.print("[bold bright_white]Human Feedback (interactive)[/bold bright_white]")
    console.print("  Enter = accept Meta suggestion")
    console.print("  k = KEEP")
    console.print("  r = REVISE")
    console.print("  a = approve all")

    for idx in sorted_ids:
        stem = stems[idx]
        meta_decision = meta_suggestions.get(idx, "REVISE")
        prefill = "KEEP" if meta_decision == "KEEP" else "REVISE"

        console.print(f"\n[bold]{idx})[/bold] {stem}")
        console.print(f"    Meta suggestion: [cyan]{meta_decision}[/cyan]")
        raw = console.input(
            f"    [bold bright_white]Decision [{prefill}] (Enter/k/r/a): [/bold bright_white]"
        ).strip().lower()

        if raw == "a":
            return "approve"
        if raw == "":
            item_decisions[idx] = prefill
        elif raw == "k":
            item_decisions[idx] = "KEEP"
        elif raw == "r":
            item_decisions[idx] = "REVISE"
        else:
            console.print("    [yellow]Invalid input, keeping default.[/yellow]")
            item_decisions[idx] = prefill

    def _render_summary() -> None:
        keep_ids = [n for n, d in item_decisions.items() if d == "KEEP"]
        revise_ids = [n for n, d in item_decisions.items() if d == "REVISE"]
        console.print(f"  KEEP: {', '.join(str(n) for n in sorted(keep_ids)) or '-'}")
        console.print(f"  REVISE: {', '.join(str(n) for n in sorted(revise_ids)) or '-'}")

    console.print("\n[bold]Selected decisions:[/bold]")
    _render_summary()
    action = (
        console.input(
            "[bold bright_white]> Submit (s), edit item number, or approve all (a): [/bold bright_white]"
        )
        .strip()
        .lower()
    )
    while True:
        if action == "a":
            return "approve"
        if action == "s":
            break
        selected = _parse_number_list(action, valid_ids)
        if not selected:
            action = (
                console.input(
                    "[bold bright_white]> Invalid input. Enter item number, s, or a: [/bold bright_white]"
                )
                .strip()
                .lower()
            )
            continue
        target = selected[0]
        toggle = (
            console.input(
                f"[bold bright_white]> Item {target} is {item_decisions[target]}. Set k/r: [/bold bright_white]"
            )
            .strip()
            .lower()
        )
        if toggle == "k":
            item_decisions[target] = "KEEP"
        elif toggle == "r":
            item_decisions[target] = "REVISE"
        else:
            console.print("[yellow]Invalid choice, unchanged.[/yellow]")
        _render_summary()
        action = (
            console.input(
                "[bold bright_white]> Submit (s), edit item number, or approve all (a): [/bold bright_white]"
            )
            .strip()
            .lower()
        )

    notes = console.input("[bold bright_white]> Extra notes (optional): [/bold bright_white]").strip()
    return {
        "approve": False,
        "item_decisions": {str(k): v for k, v in sorted(item_decisions.items())},
        "global_note": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LM-AIG: Multi-Agent Automatic Item Generation",
    )
    # Construct selection (mutually exclusive: preset or file)
    construct_group = parser.add_mutually_exclusive_group()
    construct_group.add_argument(
        "--preset",
        default="aaaw",
        choices=list_presets(),
        help="Built-in construct preset (default: aaaw).",
    )
    construct_group.add_argument(
        "--construct-file",
        "--json",
        nargs="?",
        const="examples/custom_construct.json",
        type=str,
        default=None,
        help=(
            "Path to a JSON file with custom construct definition "
            "(alias: --json). If used without value, defaults to "
            "examples/custom_construct.json."
        ),
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=None,
        help="Maximum number of revision rounds.",
    )
    parser.add_argument(
        "--lewmod",
        action="store_true",
        default=False,
        help="Use LewMod (automated LLM expert) instead of human feedback.",
    )
    parser.add_argument(
        "--verbose-json",
        action="store_true",
        default=False,
        help="Show raw JSON blocks for reviewer/meta outputs.",
    )
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    set_verbose_json_output(args.verbose_json)
    settings = get_settings()
    setup_logging(settings.log_level)
    agent_settings = get_agent_settings()
    max_revisions = (
        args.max_revisions if args.max_revisions is not None else agent_settings.workflow.max_revisions
    )

    # Resolve construct (preset or custom JSON file)
    if args.construct_file:
        construct = load_construct_from_file(args.construct_file)
    else:
        construct = get_preset(args.preset)
        if construct is None:
            console.print(f"[bold red]Unknown preset: {args.preset}[/bold red]")
            return

    # Build dimension info text (passed through state to review chain)
    dimension_info = build_dimension_info(construct)
    fingerprint = compute_fingerprint(construct)

    # Display startup info with rich console
    print_header(construct.name, agent_settings.defaults.model, max_revisions, lewmod=args.lewmod)
    print_langsmith_status(settings.langchain_tracing_v2)

    # Build the graph
    graph = build_main_workflow(lewmod=args.lewmod)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    conn = None
    run_id: str | None = None
    try:
        # Initialize persistence
        conn = get_connection()
        run_id = str(uuid.uuid4())
        create_run(
            conn,
            run_id=run_id,
            construct_name=construct.name,
            construct_definition=construct.definition,
            construct_fingerprint=fingerprint,
            mode="lewmod" if args.lewmod else "human",
            model=agent_settings.defaults.model,
            max_revisions=999 if args.lewmod else max_revisions,
        )

        db_row = conn.execute("PRAGMA database_list").fetchone()
        db_path = str(db_row["file"]) if db_row and db_row["file"] else str(DB_PATH)

        # Initial state
        initial_state = {
            "construct_name": construct.name,
            "construct_definition": construct.definition,
            "dimension_info": dimension_info,
            "construct_fingerprint": fingerprint,
            "current_phase": Phase.WEB_RESEARCH,
            "revision_count": 0,
            "max_revisions": 999 if args.lewmod else max_revisions,
            "run_id": run_id,
            "db_path": db_path,
            "previously_approved_items": [],
            "human_item_decisions": {},
            "human_global_note": "",
            "messages": [],
        }

        print_info("Starting workflow...")

        # Stream the graph execution
        # Agent messages are printed by each agent via console.print_agent_message()
        try:
            async for _event in graph.astream(initial_state, config, stream_mode="updates"):
                pass  # Agent console output is handled within each agent node
        except GraphInterrupt:
            pass

        # Check for interrupt (human-in-the-loop)
        state = graph.get_state(config)

        while state.next:
            # Get feedback from user
            console.print()
            human_input = _collect_human_feedback(state.values)
            if isinstance(human_input, dict):
                console.print("[dim]Submitting structured human decisions...[/dim]")
            else:
                console.print("[dim]Submitting approval...[/dim]")

            # Resume the graph with human feedback
            try:
                async for _event in graph.astream(
                    Command(resume=human_input), config, stream_mode="updates"
                ):
                    pass  # Agent console output is handled within each agent node
            except GraphInterrupt:
                pass  # Expected: another interrupt for the next feedback round
            except Exception as e:
                console.print(f"\n[bold red]Error during revision:[/bold red] {e}")
                console.print(
                    "[yellow]The graph state has been preserved. You can try providing feedback again.[/yellow]"
                )

            state = graph.get_state(config)

        # Print final results
        final_state = graph.get_state(config).values
        items_text = final_state.get("items_text", "No items generated.")
        review_text = final_state.get("review_text", "")
        keep_metrics_section = _build_keep_metrics_section(
            items_text,
            review_text,
            frozen_item_numbers=final_state.get("frozen_item_numbers", []),
            conn=conn,
            run_id=run_id,
        )
        review_display = review_text
        if review_text and not args.verbose_json:
            try:
                parsed_meta = MetaEditorOutput.model_validate(parse_json_markdown(review_text))
                review_display = format_structured_agent_output("MetaEditor", parsed_meta)
            except Exception:
                review_display = review_text
        elif review_text and args.verbose_json:
            try:
                parsed_json = parse_json_markdown(review_text)
                review_display = json.dumps(parsed_json, ensure_ascii=True, indent=2)
            except Exception:
                review_display = review_text

        final_output = f"## Final Items\n\n{items_text}"
        if review_display:
            final_output += f"\n\n---\n\n## Final Review\n\n{review_display}"
        if keep_metrics_section:
            final_output += f"\n\n---\n\n{keep_metrics_section}"
        final_output += f"\n\n---\n\nRevision rounds: {final_state.get('revision_count', 0)}"

        print_final_results(final_output)

        # Finalize persistence
        finish_run(conn, run_id, status="done", total_revisions=final_state.get("revision_count", 0))
    except Exception:
        if conn is not None and run_id is not None:
            finish_run(conn, run_id, status="failed")
        raise
    finally:
        if conn is not None:
            conn.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
