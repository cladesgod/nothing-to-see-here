"""Tests for constructs and state schemas."""

from __future__ import annotations

import json

import pytest

from src.schemas.constructs import (
    AAAW_CONSTRUCT,
    Construct,
    ConstructDimension,
    build_dimension_info,
    compute_fingerprint,
    get_preset,
    list_presets,
    load_construct_from_file,
)
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


class TestMainStateDimensionInfo:
    """Test that MainState accepts the new dimension_info field."""

    def test_accepts_dimension_info(self):
        state: MainState = {
            "construct_name": "Test",
            "dimension_info": "formatted dimension text",
            "current_phase": "web_research",
        }
        assert state["dimension_info"] == "formatted dimension text"


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


# ---------------------------------------------------------------------------
# Construct Presets
# ---------------------------------------------------------------------------


class TestConstructPresets:
    """Test the built-in construct preset registry."""

    def test_aaaw_preset_exists(self):
        assert get_preset("aaaw") is not None

    def test_preset_returns_correct_construct(self):
        preset = get_preset("aaaw")
        assert preset is AAAW_CONSTRUCT

    def test_preset_lookup_case_insensitive(self):
        assert get_preset("AAAW") is not None
        assert get_preset("Aaaw") is not None

    def test_unknown_preset_returns_none(self):
        assert get_preset("nonexistent") is None

    def test_list_presets_includes_aaaw(self):
        presets = list_presets()
        assert "aaaw" in presets

    def test_list_presets_returns_list(self):
        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) >= 1


# ---------------------------------------------------------------------------
# build_dimension_info
# ---------------------------------------------------------------------------


class TestBuildDimensionInfo:
    """Test the dimension info builder function."""

    def test_returns_nonempty_string(self):
        info = build_dimension_info(AAAW_CONSTRUCT)
        assert len(info) > 0

    def test_contains_all_dimension_names(self):
        info = build_dimension_info(AAAW_CONSTRUCT)
        for dim in AAAW_CONSTRUCT.dimensions:
            assert dim.name in info

    def test_contains_target_and_orbiting_labels(self):
        info = build_dimension_info(AAAW_CONSTRUCT)
        assert "TARGET" in info
        assert "ORBITING" in info

    def test_works_with_minimal_construct(self):
        minimal = Construct(
            name="Test",
            definition="A test construct.",
            dimensions=[
                ConstructDimension(
                    name="Dim1",
                    definition="Dim1 definition.",
                    orbiting_dimensions=[],
                )
            ],
        )
        info = build_dimension_info(minimal)
        assert "Dim1" in info
        # No orbiting labels for single dimension without orbiting
        assert "ORBITING" not in info

    def test_works_with_two_orbiting(self):
        construct = Construct(
            name="Multi",
            definition="A multi-dimensional construct.",
            dimensions=[
                ConstructDimension(
                    name="A",
                    definition="Dimension A.",
                    orbiting_dimensions=["B", "C"],
                ),
                ConstructDimension(
                    name="B",
                    definition="Dimension B.",
                    orbiting_dimensions=["A", "C"],
                ),
                ConstructDimension(
                    name="C",
                    definition="Dimension C.",
                    orbiting_dimensions=["A", "B"],
                ),
            ],
        )
        info = build_dimension_info(construct)
        assert "Dimension A" in info
        assert "Dimension B" in info
        assert "ORBITING" in info


# ---------------------------------------------------------------------------
# load_construct_from_file
# ---------------------------------------------------------------------------


class TestLoadConstructFromFile:
    """Test loading constructs from JSON files."""

    def test_loads_valid_json(self, tmp_path):
        construct_data = {
            "name": "Test Construct",
            "definition": "A test.",
            "dimensions": [
                {
                    "name": "Dim1",
                    "definition": "Dim1 definition.",
                    "example_items": ["Example item 1"],
                    "orbiting_dimensions": [],
                }
            ],
        }
        path = tmp_path / "test_construct.json"
        path.write_text(json.dumps(construct_data))
        construct = load_construct_from_file(path)
        assert construct.name == "Test Construct"
        assert len(construct.dimensions) == 1
        assert construct.dimensions[0].name == "Dim1"

    def test_loads_multi_dimension_json(self, tmp_path):
        construct_data = {
            "name": "Multi",
            "definition": "Multi-dim.",
            "dimensions": [
                {
                    "name": "A",
                    "definition": "Def A.",
                    "orbiting_dimensions": ["B"],
                },
                {
                    "name": "B",
                    "definition": "Def B.",
                    "orbiting_dimensions": ["A"],
                },
            ],
        }
        path = tmp_path / "multi.json"
        path.write_text(json.dumps(construct_data))
        construct = load_construct_from_file(path)
        assert len(construct.dimensions) == 2

    def test_validates_missing_required_fields(self, tmp_path):
        data = {"name": "X", "definition": "Y", "dimensions": [{"name": "D1"}]}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data))
        with pytest.raises(Exception):  # pydantic.ValidationError
            load_construct_from_file(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_construct_from_file("/nonexistent/path.json")


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------


class TestComputeFingerprint:
    """Test SHA-256 construct fingerprinting for memory correctness."""

    def test_same_construct_same_hash(self):
        fp1 = compute_fingerprint(AAAW_CONSTRUCT)
        fp2 = compute_fingerprint(AAAW_CONSTRUCT)
        assert fp1 == fp2

    def test_deterministic(self):
        """Multiple calls produce identical results."""
        results = {compute_fingerprint(AAAW_CONSTRUCT) for _ in range(10)}
        assert len(results) == 1

    def test_returns_hex_string(self):
        fp = compute_fingerprint(AAAW_CONSTRUCT)
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA-256 hex digest
        int(fp, 16)  # Should parse as hex without error

    def test_different_name_different_hash(self):
        c1 = Construct(
            name="Construct A",
            definition="Same definition.",
            dimensions=[
                ConstructDimension(name="D1", definition="Def.", orbiting_dimensions=[]),
            ],
        )
        c2 = Construct(
            name="Construct B",
            definition="Same definition.",
            dimensions=[
                ConstructDimension(name="D1", definition="Def.", orbiting_dimensions=[]),
            ],
        )
        assert compute_fingerprint(c1) != compute_fingerprint(c2)

    def test_different_dimension_different_hash(self):
        c1 = Construct(
            name="Test",
            definition="Def.",
            dimensions=[
                ConstructDimension(name="DimA", definition="Def A.", orbiting_dimensions=[]),
            ],
        )
        c2 = Construct(
            name="Test",
            definition="Def.",
            dimensions=[
                ConstructDimension(name="DimB", definition="Def B.", orbiting_dimensions=[]),
            ],
        )
        assert compute_fingerprint(c1) != compute_fingerprint(c2)
