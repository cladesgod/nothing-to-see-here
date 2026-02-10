"""Golden dataset and LangSmith dataset management.

Golden dataset contains the 24 LM-AIG generated items from Lee et al. (2025)
Table 3, used as a benchmark for evaluating our pipeline's output quality.
"""

from __future__ import annotations

import structlog

from src.schemas.constructs import AAAW_CONSTRUCT

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Golden Dataset — Paper Table 3 (Lee et al., 2025)
# 4 items per dimension × 6 dimensions = 24 items
# ---------------------------------------------------------------------------

GOLDEN_ITEMS: dict[str, list[str]] = {
    "AI Use Anxiety": [
        "I feel uneasy about using AI tools during my work tasks.",
        "The possibility of making mistakes when using AI makes me feel anxious at work.",
        "I am worried that I don't have the necessary skills to use AI at work effectively.",
        "I feel apprehensive about how AI could change my role at work.",
    ],
    "Personal Utility": [
        "AI tools enhance my efficiency in completing work tasks.",
        "Using AI increases my confidence in handling work challenges.",
        "AI streamlines routine tasks, allowing me to focus on important work.",
        "The learning resources provided by AI aid my career development.",
    ],
    "Perceived Humanlikeness of AI": [
        "AI systems seem capable of expressing emotions.",
        "AI systems act in ways that resemble human behavior.",
        "Interacting with AI feels natural and human-like.",
        "AI displays empathy similar to humans.",
    ],
    "Perceived Adaptability of AI": [
        "AI in the workplace can learn and improve its functions.",
        "AI systems at work can adjust to new tasks.",
        "AI can develop new skills when facing work challenges.",
        "AI's learning capacity enhances efficiency at work.",
    ],
    "Perceived Quality of AI": [
        "AI consistently provides accurate information at work.",
        "AI provides reliable data for my tasks.",
        "AI presents information in a clear format.",
        "AI applications function consistently without errors.",
    ],
    "Job Insecurity": [
        "I think AI could replace my role at work.",
        "I believe my job could be replaced by AI in the near future.",
        "I feel my job security is at risk because of AI.",
        "I worry that AI could perform my tasks better than I can.",
    ],
}


def get_golden_dataset() -> list[dict]:
    """Return the golden dataset as a list of evaluation examples.

    Each example contains:
        - item_text: The Likert-scale item
        - dimension_name: Target AAAW dimension
        - dimension_definition: Definition of the dimension
        - orbiting_dimensions: List of (name, definition) tuples
    """
    examples = []
    for dimension_name, items in GOLDEN_ITEMS.items():
        dim = AAAW_CONSTRUCT.get_dimension(dimension_name)
        if dim is None:
            continue

        orbiting = AAAW_CONSTRUCT.get_orbiting_definitions(dimension_name)

        for item_text in items:
            examples.append({
                "item_text": item_text,
                "dimension_name": dimension_name,
                "dimension_definition": dim.definition,
                "orbiting_dimensions": orbiting,
            })

    return examples


def parse_items_from_text(items_text: str) -> list[str]:
    """Parse numbered items from plain text output.

    Handles formats like:
        1. Item text here.
        2. Another item text.
    """
    items = []
    for line in items_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading number + period/parenthesis
        for i, ch in enumerate(line):
            if ch in ".)" and i > 0 and line[:i].strip().isdigit():
                item = line[i + 1 :].strip()
                if item:
                    items.append(item)
                break
    return items


def create_langsmith_dataset(dataset_name: str = "lm-aig-golden") -> None:
    """Create a LangSmith dataset from the golden items.

    Requires LANGCHAIN_API_KEY to be set.
    """
    try:
        from langsmith import Client
    except ImportError:
        logger.error("langsmith not installed — run: pip install langsmith")
        return

    client = Client()

    # Check if dataset already exists
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        logger.info("dataset_exists", name=dataset_name, id=str(existing.id))
        return
    except Exception:
        pass  # Dataset doesn't exist, create it

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Golden dataset from Lee et al. (2025) Table 3 — 24 AAAW items",
    )

    examples = get_golden_dataset()
    for ex in examples:
        client.create_example(
            dataset_id=dataset.id,
            inputs={
                "item_text": ex["item_text"],
                "dimension_name": ex["dimension_name"],
                "dimension_definition": ex["dimension_definition"],
            },
            outputs={
                "expected_quality": "high",  # Golden items are the benchmark
            },
            metadata={
                "source": "paper_table_3",
                "construct": "AAAW",
            },
        )

    logger.info("dataset_created", name=dataset_name, examples=len(examples))
