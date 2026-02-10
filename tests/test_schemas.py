"""Tests for constructs and state schemas."""

from __future__ import annotations

from src.schemas.constructs import AAAW_CONSTRUCT
from src.schemas.state import MainState, ReviewChainState


class TestAAWConstruct:
    """Test the AAAW construct definition and structure."""

    def test_construct_has_6_dimensions(self):
        assert len(AAAW_CONSTRUCT.dimensions) == 6

    def test_construct_name_contains_ai_workplace(self):
        assert "AI in the Workplace" in AAAW_CONSTRUCT.name

    def test_dimension_names(self):
        names = {d.name for d in AAAW_CONSTRUCT.dimensions}
        expected = {
            "AI Use Anxiety",
            "Personal Utility",
            "Perceived Humanlikeness of AI",
            "Perceived Adaptability of AI",
            "Perceived Quality of AI",
            "Job Insecurity",
        }
        assert names == expected

    def test_each_dimension_has_2_orbiting(self):
        for dim in AAAW_CONSTRUCT.dimensions:
            assert len(dim.orbiting_dimensions) == 2, (
                f"{dim.name} should have 2 orbiting dimensions"
            )

    def test_orbiting_dimensions_are_valid_dimension_names(self):
        dim_names = {d.name for d in AAAW_CONSTRUCT.dimensions}
        for dim in AAAW_CONSTRUCT.dimensions:
            for orb in dim.orbiting_dimensions:
                assert orb in dim_names, (
                    f"Orbiting '{orb}' of '{dim.name}' not in construct"
                )

    def test_get_orbiting_definitions_returns_pairs(self):
        orbiting = AAAW_CONSTRUCT.get_orbiting_definitions("AI Use Anxiety")
        assert len(orbiting) == 2
        names = [o[0] for o in orbiting]
        assert "Job Insecurity" in names
        assert "Personal Utility" in names

    def test_get_dimension_found(self):
        dim = AAAW_CONSTRUCT.get_dimension("Job Insecurity")
        assert dim is not None
        assert dim.name == "Job Insecurity"

    def test_get_dimension_not_found(self):
        assert AAAW_CONSTRUCT.get_dimension("Nonexistent") is None

    def test_each_dimension_has_definition(self):
        for dim in AAAW_CONSTRUCT.dimensions:
            assert dim.definition, f"{dim.name} has empty definition"


class TestMainState:
    """Tests for the MainState TypedDict."""

    def test_accepts_all_text_fields(self):
        state: MainState = {
            "construct_name": "Test",
            "construct_definition": "Def",
            "current_phase": "web_research",
            "items_text": "1. Item",
            "review_text": "Pass",
            "messages": ["msg"],
        }
        assert state["items_text"] == "1. Item"

    def test_minimal_state(self):
        state: MainState = {"current_phase": "web_research"}
        assert state["current_phase"] == "web_research"


class TestReviewChainState:
    """Tests for the ReviewChainState TypedDict."""

    def test_accepts_review_fields(self):
        state: ReviewChainState = {
            "items_text": "items",
            "construct_name": "AAAW",
            "dimension_info": "info",
            "content_review": "content",
            "linguistic_review": "linguistic",
            "bias_review": "bias",
            "meta_review": "meta",
        }
        assert state["meta_review"] == "meta"
