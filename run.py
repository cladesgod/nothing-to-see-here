"""CLI entry point for the LM-AIG Multi-Agent System.

Usage:
    python run.py                           # Run with human feedback (default)
    python run.py --lewmod                  # Run with LewMod (automated feedback)
    python run.py --construct "Social Adjustment"  # Run with specific dimension
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from src.config import get_agent_settings, get_settings
from src.graphs.main_workflow import build_main_workflow
from src.logging_config import setup_logging
from src.persistence.db import get_connection
from src.persistence.repository import create_run, finish_run
from src.schemas.constructs import AAAW_CONSTRUCT
from src.utils.console import (
    console,
    print_final_results,
    print_header,
    print_info,
    print_langsmith_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LM-AIG: Multi-Agent Automatic Item Generation",
    )
    parser.add_argument(
        "--construct",
        default=AAAW_CONSTRUCT.name,
        help="Name of the target construct.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=get_agent_settings().workflow.max_revisions,
        help="Maximum number of revision rounds.",
    )
    parser.add_argument(
        "--lewmod",
        action="store_true",
        default=False,
        help="Use LewMod (automated LLM expert) instead of human feedback.",
    )
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    # Display startup info with rich console
    print_header(args.construct, get_agent_settings().defaults.model, args.max_revisions, lewmod=args.lewmod)
    print_langsmith_status(settings.langchain_tracing_v2)

    # Build the graph
    graph = build_main_workflow(lewmod=args.lewmod)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Initialize persistence
    conn = get_connection()
    run_id = str(uuid.uuid4())
    agent_settings = get_agent_settings()
    create_run(
        conn,
        run_id=run_id,
        construct_name=AAAW_CONSTRUCT.name,
        construct_definition=AAAW_CONSTRUCT.definition,
        mode="lewmod" if args.lewmod else "human",
        model=agent_settings.defaults.model,
        max_revisions=999 if args.lewmod else args.max_revisions,
    )

    # Initial state
    initial_state = {
        "construct_name": AAAW_CONSTRUCT.name,
        "construct_definition": AAAW_CONSTRUCT.definition,
        "current_phase": "web_research",
        "revision_count": 0,
        "max_revisions": 999 if args.lewmod else args.max_revisions,
        "run_id": run_id,
        "db_path": str(conn.execute("PRAGMA database_list").fetchone()[2]),
        "previously_approved_items": [],
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
        human_input = console.input("[bold bright_white]> Your feedback: [/bold bright_white]").strip()

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
            console.print("[yellow]The graph state has been preserved. You can try providing feedback again.[/yellow]")

        state = graph.get_state(config)

    # Print final results
    final_state = graph.get_state(config).values
    items_text = final_state.get("items_text", "No items generated.")
    review_text = final_state.get("review_text", "")

    final_output = f"## Final Items\n\n{items_text}"
    if review_text:
        final_output += f"\n\n---\n\n## Final Review\n\n{review_text}"
    final_output += f"\n\n---\n\nRevision rounds: {final_state.get('revision_count', 0)}"

    print_final_results(final_output)

    # Finalize persistence
    finish_run(conn, run_id, status="done", total_revisions=final_state.get("revision_count", 0))
    conn.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
