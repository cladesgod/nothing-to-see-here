from __future__ import annotations

from run import _build_keep_metrics_section, _extract_metrics_from_reason


def test_extract_metrics_from_reason_parses_values():
    reason = (
        "content(c=0.83, d=0.42, ok=True); ling_min=4; bias=5; content_note=good"
    )
    metrics = _extract_metrics_from_reason(reason)
    assert metrics == {"c": "0.83", "d": "0.42", "ling_min": "4", "bias": "5"}


def test_build_keep_metrics_section_lists_keep_items_and_metrics():
    items_text = (
        "1. Item one stem. Rationale: x\n"
        "2. Item two stem. Rationale: y\n"
        "\n"
        "Response scale: 1 to 7"
    )
    review_text = (
        '{"items":['
        '{"item_number":1,"decision":"KEEP","reason":"content(c=0.90, d=0.40, ok=True); ling_min=4; bias=5","revised_item_stem":null},'
        '{"item_number":2,"decision":"REVISE","reason":"content(c=0.70, d=0.20, ok=False); ling_min=3; bias=4","revised_item_stem":"alt"}'
        '],"overall_synthesis":"x"}'
    )

    section = _build_keep_metrics_section(items_text, review_text)
    assert "## Final KEEP Items & Metrics" in section
    assert "Item 1: Item one stem. Rationale: x" in section
    assert "c=0.90, d=0.40, ling_min=4, bias=5" in section
    assert "Item 2" not in section


def test_build_keep_metrics_section_no_keep_message():
    items_text = "1. Item one. Rationale: x"
    review_text = (
        '{"items":['
        '{"item_number":1,"decision":"REVISE","reason":"content(c=0.70, d=0.20, ok=False); ling_min=3; bias=4","revised_item_stem":"alt"}'
        '],"overall_synthesis":"x"}'
    )
    section = _build_keep_metrics_section(items_text, review_text)
    assert "No KEEP items found in final review or frozen-history for this run." in section


def test_build_keep_metrics_section_includes_frozen_keep_items_with_fallback_metrics():
    items_text = (
        "1. Item one stem. Rationale: x\n"
        "2. Item two stem. Rationale: y\n"
        "3. Item three stem. Rationale: z"
    )
    section = _build_keep_metrics_section(
        items_text,
        review_text="",
        frozen_item_numbers=[2, 3],
    )
    assert "Item 2: Item two stem. Rationale: y" in section
    assert "Item 3: Item three stem. Rationale: z" in section
    assert "c=-, d=-, ling_min=-, bias=-" in section
    assert "deterministic metrics were not found" in section

