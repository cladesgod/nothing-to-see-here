"""Tests for constructs and state schemas."""

from __future__ import annotations

from src.schemas.constructs import AAAW_CONSTRUCT
from src.schemas.state import MainState, ReviewChainState


class TestAAWConstruct:
    def test_construct_has_dimensions(self):
        assert len(AAAW_CONSTRUCT.dimensions) == 6

    def test_dimension_names(self):
        names = [d.name for d in AAAW_CONSTRUCT.dimensions]
        assert "AI Use Anxiety" in names
        assert "Personal Utility" in names
        assert "Perceived Humanlikeness of AI" in names
        assert "Perceived Adaptability of AI" in names
        assert "Perceived Quality of AI" in names
        assert "Job Insecurity" in names

    def test_construct_name(self):
        assert "AI in the Workplace" in AAAW_CONSTRUCT.name

    def test_each_dimension_has_orbiting(self):
        for dim in AAAW_CONSTRUCT.dimensions:
            assert len(dim.orbiting_dimensions) == 2, (
                f"{dim.name} should have 2 orbiting dimensions"
            )

    def test_orbiting_dimensions_exist(self):
        dim_names = {d.name for d in AAAW_CONSTRUCT.dimensions}
        for dim in AAAW_CONSTRUCT.dimensions:
            for orb in dim.orbiting_dimensions:
                assert orb in dim_names, (
                    f"Orbiting dimension '{orb}' of '{dim.name}' not found"
                )

    def test_get_orbiting_definitions(self):
        orbiting = AAAW_CONSTRUCT.get_orbiting_definitions("AI Use Anxiety")
        assert len(orbiting) == 2
        names = [o[0] for o in orbiting]
        assert "Job Insecurity" in names
        assert "Personal Utility" in names

    def test_get_dimension(self):
        dim = AAAW_CONSTRUCT.get_dimension("Job Insecurity")
        assert dim is not None
        assert "job" in dim.definition.lower() or "threat" in dim.definition.lower()

    def test_get_dimension_missing(self):
        assert AAAW_CONSTRUCT.get_dimension("Nonexistent") is None


class TestMainState:
    """Tests for the MainState TypedDict structure."""

    def test_main_state_accepts_text_fields(self):
        state: MainState = {
            "construct_name": "Test Construct",
            "construct_definition": "A test definition.",
            "current_phase": "web_research",
            "items_text": "1. Item one\n2. Item two",
            "review_text": "All items pass.",
            "messages": ["[Critic] Routing"],
        }
        assert state["items_text"] == "1. Item one\n2. Item two"
        assert state["review_text"] == "All items pass."

    def test_main_state_minimal(self):
        state: MainState = {"current_phase": "web_research"}
        assert state["current_phase"] == "web_research"


class TestReviewChainState:
    """Tests for the ReviewChainState TypedDict structure."""

    def test_review_chain_state_text_fields(self):
        state: ReviewChainState = {
            "items_text": "1. Test item",
            "construct_name": "AAAW",
            "dimension_info": "Target: AI Use Anxiety",
            "content_review": "Items are relevant.",
            "linguistic_review": "Grammar is correct.",
            "bias_review": "No bias detected.",
            "meta_review": "All items KEEP.",
        }
        assert state["content_review"] == "Items are relevant."
        assert state["meta_review"] == "All items KEEP."
