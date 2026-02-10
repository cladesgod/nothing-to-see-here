"""Graph state definitions using TypedDict.

MainState: outer graph state (full workflow).
ReviewChainState: inner subgraph state (review pipeline).

All agent communication uses natural language text (paper-like approach).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ReviewChainState(TypedDict, total=False):
    """State for the inner review-chain subgraph.

    Flows: [content, linguistic, bias] (parallel) → meta_editor
    All fields are natural language text strings.
    """

    # The items being reviewed (natural language text from Item Writer)
    items_text: str

    # Construct/dimension context for reviewers
    construct_name: str
    construct_definition: str
    dimension_info: str  # formatted dimension + orbiting info for content reviewer

    # Review outputs (natural language text from each reviewer)
    content_review: str
    linguistic_review: str
    bias_review: str

    # Final synthesis by the meta editor (natural language text)
    meta_review: str


class MainState(TypedDict, total=False):
    """State for the outer workflow graph.

    Tracks the full lifecycle: research → write → review → human → revise.
    All agent outputs are natural language text.
    """

    # ----- Input -----
    construct_name: str
    construct_definition: str

    # ----- Phase tracking -----
    current_phase: str  # web_research | item_generation | review | human_feedback | revision | done

    # ----- WebSurfer output -----
    research_summary: str

    # ----- Item Writer output (natural language text) -----
    items_text: str

    # ----- Review Chain output (natural language text from Meta Editor) -----
    review_text: str

    # ----- Human feedback -----
    human_feedback: str

    # ----- Iteration control -----
    revision_count: int
    max_revisions: int  # default 3 (from paper)

    # ----- Persistence -----
    run_id: str      # Current run UUID (for DB tracking)
    db_path: str     # SQLite DB path (serializable, not a connection)

    # ----- Item diversity (avoids cross-round/cross-run homogeneity) -----
    previously_approved_items: Annotated[list[str], operator.add]

    # ----- Messages (for debugging / logging — these DO accumulate) -----
    messages: Annotated[list[str], operator.add]
