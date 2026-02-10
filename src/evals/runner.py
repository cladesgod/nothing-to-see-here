"""Offline evaluation runner.

Orchestrates running all evaluators on a dataset (golden or custom items)
and produces a quality report with aggregate metrics.
"""

from __future__ import annotations

import asyncio

import structlog
from rich.console import Console
from rich.table import Table

from src.evals.dataset import get_golden_dataset, parse_items_from_text
from src.evals.evaluators import (
    bias_evaluator,
    content_validity_evaluator,
    linguistic_quality_evaluator,
    overall_quality_evaluator,
)
from src.schemas.constructs import AAAW_CONSTRUCT

logger = structlog.get_logger(__name__)
console = Console()


async def evaluate_single_item(example: dict) -> dict:
    """Run all 4 evaluators on a single item and return results."""
    item_text = example["item_text"]
    dimension_name = example["dimension_name"]
    dimension_definition = example["dimension_definition"]
    orbiting_dimensions = example["orbiting_dimensions"]

    # Run 3 independent evaluators in parallel
    content_task = content_validity_evaluator(
        item_text=item_text,
        dimension_name=dimension_name,
        dimension_definition=dimension_definition,
        orbiting_dimensions=orbiting_dimensions,
    )
    linguistic_task = linguistic_quality_evaluator(item_text=item_text)
    bias_task = bias_evaluator(item_text=item_text)

    content_result, linguistic_result, bias_result = await asyncio.gather(
        content_task, linguistic_task, bias_task
    )

    # Run overall evaluator with individual scores
    overall_result = await overall_quality_evaluator(
        item_text=item_text,
        dimension_name=dimension_name,
        content_score=content_result["score"],
        linguistic_score=linguistic_result["score"],
        bias_score=bias_result["score"],
    )

    return {
        "item_text": item_text,
        "dimension": dimension_name,
        "content": content_result,
        "linguistic": linguistic_result,
        "bias": bias_result,
        "overall": overall_result,
    }


async def run_golden_eval() -> list[dict]:
    """Run evaluation on the golden dataset (paper Table 3)."""
    examples = get_golden_dataset()
    console.print(f"\n[bold]Evaluating {len(examples)} golden items...[/bold]\n")

    results = []
    for i, example in enumerate(examples, 1):
        console.print(
            f"  [{i}/{len(examples)}] {example['dimension_name']}: "
            f"{example['item_text'][:60]}..."
        )
        result = await evaluate_single_item(example)
        results.append(result)

    return results


async def run_custom_eval(items_text: str, dimension_name: str) -> list[dict]:
    """Run evaluation on custom items (e.g., from a pipeline run).

    Args:
        items_text: Plain text with numbered items.
        dimension_name: Target AAAW dimension name.
    """
    items = parse_items_from_text(items_text)
    if not items:
        console.print("[red]No items found in text.[/red]")
        return []

    dim = AAAW_CONSTRUCT.get_dimension(dimension_name)
    if dim is None:
        console.print(f"[red]Unknown dimension: {dimension_name}[/red]")
        return []

    orbiting = AAAW_CONSTRUCT.get_orbiting_definitions(dimension_name)

    examples = [
        {
            "item_text": item,
            "dimension_name": dimension_name,
            "dimension_definition": dim.definition,
            "orbiting_dimensions": orbiting,
        }
        for item in items
    ]

    console.print(f"\n[bold]Evaluating {len(examples)} items for {dimension_name}...[/bold]\n")

    results = []
    for i, example in enumerate(examples, 1):
        console.print(f"  [{i}/{len(examples)}] {example['item_text'][:60]}...")
        result = await evaluate_single_item(example)
        results.append(result)

    return results


def print_eval_report(results: list[dict]) -> None:
    """Print a formatted evaluation report using rich."""
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    # Build results table
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Item", style="cyan", max_width=50)
    table.add_column("Dimension", style="magenta")
    table.add_column("Content", justify="center")
    table.add_column("Linguistic", justify="center")
    table.add_column("Bias", justify="center")
    table.add_column("Overall", justify="center")
    table.add_column("Decision", justify="center")

    for r in results:
        overall_score = r["overall"]["score"]
        if overall_score >= 0.8:
            decision = "[green]ACCEPT[/green]"
        elif overall_score >= 0.5:
            decision = "[yellow]REVISE[/yellow]"
        else:
            decision = "[red]REJECT[/red]"

        def _fmt(score: float) -> str:
            color = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
            return f"[{color}]{score:.2f}[/{color}]"

        table.add_row(
            r["item_text"][:50] + ("..." if len(r["item_text"]) > 50 else ""),
            r["dimension"],
            _fmt(r["content"]["score"]),
            _fmt(r["linguistic"]["score"]),
            _fmt(r["bias"]["score"]),
            _fmt(r["overall"]["score"]),
            decision,
        )

    console.print(table)

    # Aggregate metrics
    total = len(results)
    avg_content = sum(r["content"]["score"] for r in results) / total
    avg_linguistic = sum(r["linguistic"]["score"] for r in results) / total
    avg_bias = sum(r["bias"]["score"] for r in results) / total
    avg_overall = sum(r["overall"]["score"] for r in results) / total
    accept_count = sum(1 for r in results if r["overall"]["score"] >= 0.8)
    revise_count = sum(1 for r in results if 0.5 <= r["overall"]["score"] < 0.8)
    reject_count = sum(1 for r in results if r["overall"]["score"] < 0.5)

    console.print("\n[bold]Aggregate Metrics:[/bold]")
    console.print(f"  Items evaluated: {total}")
    console.print(f"  Avg Content Validity: {avg_content:.2f}")
    console.print(f"  Avg Linguistic Quality: {avg_linguistic:.2f}")
    console.print(f"  Avg Bias Freedom: {avg_bias:.2f}")
    console.print(f"  Avg Overall Quality: {avg_overall:.2f}")
    console.print(f"  Decisions: [green]{accept_count} ACCEPT[/green] | "
                  f"[yellow]{revise_count} REVISE[/yellow] | "
                  f"[red]{reject_count} REJECT[/red]")
    console.print(f"  Pass Rate: {accept_count / total * 100:.0f}%\n")
