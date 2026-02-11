"""Tests for prompt template formatting safety."""

from src.prompts.templates import (
    BIAS_REVIEWER_TASK,
    CONTENT_REVIEWER_TASK,
    ITEM_WRITER_REVISE,
    LEWMOD_TASK,
    LINGUISTIC_REVIEWER_TASK,
    META_EDITOR_TASK,
)
from run import _parse_number_list, _parse_numbered_item_stems


def test_meta_editor_task_format_does_not_raise_keyerror():
    text = META_EDITOR_TASK.format(
        items_text="1. Item",
        content_review="content",
        linguistic_review="ling",
        bias_review="bias",
    )
    assert "Return ONLY JSON" in text


def test_item_writer_revise_format_has_construct_context():
    text = ITEM_WRITER_REVISE.format(
        construct_name="AAAW",
        construct_definition="Attitudes toward AI",
        items_text="1. Item",
        review_text="review",
        human_feedback="feedback",
    )
    assert "AAAW" in text
    assert "Attitudes toward AI" in text
    assert "decision" in text  # schema explanation present


def test_reviewer_tasks_enforce_scope_and_numbering_rules():
    content = CONTENT_REVIEWER_TASK.format(items_text="1. Item", dimension_info="dims")
    linguistic = LINGUISTIC_REVIEWER_TASK.format(items_text="1. Item", construct_name="AAAW")
    bias = BIAS_REVIEWER_TASK.format(
        items_text="1. Item",
        construct_name="AAAW",
        target_population="workers",
    )
    for task in (content, linguistic, bias):
        assert "Evaluate ONLY the items listed above." in task
        assert "Preserve original `item_number` values exactly" in task


def test_lewmod_task_has_active_item_contract():
    task = LEWMOD_TASK.format(items_text="1. Item", review_text="meta", revision_count=1)
    assert "Active-item contract" in task
    assert "Do NOT reference frozen/unlisted item numbers." in task


def test_parse_numbered_item_stems():
    stems = _parse_numbered_item_stems("1. Alpha\n2) Beta\nNot item\n3. Gamma")
    assert stems == {1: "Alpha", 2: "Beta", 3: "Gamma"}


def test_parse_number_list_filters_invalid_ids():
    nums = _parse_number_list("1, 2, 99 and 3", {1, 2, 3})
    assert nums == [1, 2, 3]
