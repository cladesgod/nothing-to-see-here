"""Workflow phase definitions shared across graph nodes."""

from __future__ import annotations

from enum import StrEnum


class Phase(StrEnum):
    """Canonical phase values for the main workflow."""

    WEB_RESEARCH = "web_research"
    ITEM_GENERATION = "item_generation"
    REVIEW = "review"
    HUMAN_FEEDBACK = "human_feedback"
    REVISION = "revision"
    DONE = "done"
