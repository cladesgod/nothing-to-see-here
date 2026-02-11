"""Tests for deterministic scoring and decision rules."""

from src.utils.deterministic_scoring import build_deterministic_meta_review


def test_keep_when_all_thresholds_pass():
    content = """
{"items":[{"item_number":1,"target_rating":6,"orbiting_1_rating":2,"orbiting_2_rating":2,"feedback":""}]}
"""
    linguistic = """
{"items":[{"item_number":1,"grammatical_accuracy":5,"ease_of_understanding":5,"negative_language_free":5,"clarity_directness":5,"feedback":""}]}
"""
    bias = """
{"items":[{"item_number":1,"score":5,"feedback":""}]}
"""
    meta = """
{"items":[{"item_number":1,"decision":"REVISE","reason":"x","revised_item_stem":"Alt stem"}],"overall_synthesis":"x"}
"""
    out = build_deterministic_meta_review(
        content_review_text=content,
        linguistic_review_text=linguistic,
        bias_review_text=bias,
        meta_review_text=meta,
    )
    assert out.items[0].decision == "KEEP"
    assert out.items[0].revised_item_stem is None


def test_revise_when_content_fails_but_not_discard_level():
    content = """
{"items":[{"item_number":1,"target_rating":5,"orbiting_1_rating":4,"orbiting_2_rating":4,"feedback":"weak distinctiveness"}]}
"""
    linguistic = """
{"items":[{"item_number":1,"grammatical_accuracy":4,"ease_of_understanding":4,"negative_language_free":4,"clarity_directness":4,"feedback":""}]}
"""
    bias = """
{"items":[{"item_number":1,"score":4,"feedback":""}]}
"""
    meta = """
{"items":[{"item_number":1,"decision":"REVISE","reason":"x","revised_item_stem":"Reworded"}],"overall_synthesis":"x"}
"""
    out = build_deterministic_meta_review(
        content_review_text=content,
        linguistic_review_text=linguistic,
        bias_review_text=bias,
        meta_review_text=meta,
    )
    assert out.items[0].decision == "REVISE"
    assert out.items[0].revised_item_stem == "Reworded"


def test_discard_when_bias_or_linguistic_severe():
    content = """
{"items":[{"item_number":1,"target_rating":6,"orbiting_1_rating":3,"orbiting_2_rating":3,"feedback":""}]}
"""
    linguistic = """
{"items":[{"item_number":1,"grammatical_accuracy":2,"ease_of_understanding":2,"negative_language_free":2,"clarity_directness":2,"feedback":""}]}
"""
    bias = """
{"items":[{"item_number":1,"score":2,"feedback":"high risk"}]}
"""
    meta = """
{"items":[{"item_number":1,"decision":"KEEP","reason":"x","revised_item_stem":null}],"overall_synthesis":"x"}
"""
    out = build_deterministic_meta_review(
        content_review_text=content,
        linguistic_review_text=linguistic,
        bias_review_text=bias,
        meta_review_text=meta,
    )
    assert out.items[0].decision == "DISCARD"


def test_low_content_alone_is_revise_not_discard():
    content = """
{"items":[{"item_number":1,"target_rating":2,"orbiting_1_rating":6,"orbiting_2_rating":6,"feedback":"wrong dimension"}]}
"""
    linguistic = """
{"items":[{"item_number":1,"grammatical_accuracy":5,"ease_of_understanding":5,"negative_language_free":5,"clarity_directness":5,"feedback":""}]}
"""
    bias = """
{"items":[{"item_number":1,"score":5,"feedback":""}]}
"""
    meta = """
{"items":[{"item_number":1,"decision":"DISCARD","reason":"x","revised_item_stem":null}],"overall_synthesis":"x"}
"""
    out = build_deterministic_meta_review(
        content_review_text=content,
        linguistic_review_text=linguistic,
        bias_review_text=bias,
        meta_review_text=meta,
    )
    assert out.items[0].decision == "REVISE"


def test_missing_content_item_still_gets_decision():
    content = """
{"items":[{"item_number":1,"target_rating":6,"orbiting_1_rating":2,"orbiting_2_rating":2,"feedback":""}]}
"""
    linguistic = """
{"items":[
  {"item_number":1,"grammatical_accuracy":5,"ease_of_understanding":5,"negative_language_free":5,"clarity_directness":5,"feedback":""},
  {"item_number":2,"grammatical_accuracy":4,"ease_of_understanding":4,"negative_language_free":4,"clarity_directness":4,"feedback":"ok"}
]}
"""
    bias = """
{"items":[{"item_number":2,"score":4,"feedback":"ok"}]}
"""
    meta = """
{"items":[{"item_number":2,"decision":"REVISE","reason":"x","revised_item_stem":"Candidate"}],"overall_synthesis":"x"}
"""
    out = build_deterministic_meta_review(
        content_review_text=content,
        linguistic_review_text=linguistic,
        bias_review_text=bias,
        meta_review_text=meta,
    )
    by_id = {item.item_number: item for item in out.items}
    assert 1 in by_id
    assert 2 in by_id
    assert by_id[2].decision == "REVISE"
    assert "missing_content_review" in by_id[2].reason
