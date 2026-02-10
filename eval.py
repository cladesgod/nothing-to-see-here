#!/usr/bin/env python3
"""Evaluation CLI — independent quality assessment for generated items.

Usage:
    # Evaluate the golden dataset (paper Table 3 items)
    python eval.py

    # Evaluate golden dataset for a specific dimension
    python eval.py --dimension "AI Use Anxiety"

    # Create LangSmith dataset from golden items
    python eval.py --create-dataset

    # Evaluate custom items from a file
    python eval.py --file items.txt --dimension "AI Use Anxiety"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="LM-AIG Evaluation Pipeline — Independent LLM-as-a-Judge"
    )
    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Create LangSmith dataset from golden items (paper Table 3)",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        default=None,
        help="Filter to a specific AAAW dimension (e.g., 'AI Use Anxiety')",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a text file with numbered items to evaluate",
    )

    args = parser.parse_args()

    # --- Create LangSmith Dataset ---
    if args.create_dataset:
        console.print("[bold]Creating LangSmith dataset from golden items...[/bold]")
        from src.evals.dataset import create_langsmith_dataset

        create_langsmith_dataset()
        console.print("[green]Done.[/green]")
        return

    # --- Evaluate Custom Items from File ---
    if args.file:
        if not args.dimension:
            console.print("[red]--dimension is required when using --file[/red]")
            sys.exit(1)

        with open(args.file, encoding="utf-8") as f:
            items_text = f.read()

        from src.evals.runner import print_eval_report, run_custom_eval

        results = asyncio.run(run_custom_eval(items_text, args.dimension))
        print_eval_report(results)
        return

    # --- Evaluate Golden Dataset ---
    from src.evals.dataset import get_golden_dataset
    from src.evals.runner import evaluate_single_item, print_eval_report

    examples = get_golden_dataset()

    # Filter by dimension if specified
    if args.dimension:
        examples = [ex for ex in examples if ex["dimension_name"] == args.dimension]
        if not examples:
            console.print(f"[red]No items found for dimension: {args.dimension}[/red]")
            console.print("Available dimensions:")
            for dim in {ex["dimension_name"] for ex in get_golden_dataset()}:
                console.print(f"  - {dim}")
            sys.exit(1)

    console.print(f"\n[bold]LM-AIG Evaluation Pipeline[/bold]")
    console.print(f"Items: {len(examples)} | Mode: Golden Dataset (Paper Table 3)\n")

    async def _run():
        results = []
        for i, example in enumerate(examples, 1):
            console.print(
                f"  [{i}/{len(examples)}] {example['dimension_name']}: "
                f"{example['item_text'][:60]}..."
            )
            result = await evaluate_single_item(example)
            results.append(result)
        return results

    results = asyncio.run(_run())
    print_eval_report(results)


if __name__ == "__main__":
    main()
