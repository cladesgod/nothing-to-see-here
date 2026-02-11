"""Deterministic scoring and decision rules for review outputs."""

from __future__ import annotations

from typing import Any

from langchain_core.utils.json import parse_json_markdown
from src.schemas.agent_outputs import (
    BiasReviewerOutput,
    ContentReviewerOutput,
    LinguisticReviewerOutput,
    MetaDecision,
    MetaEditorOutput,
)


def _safe_parse_json(text: str) -> dict[str, Any]:
    data = parse_json_markdown(text) if text else {}
    return data if isinstance(data, dict) else {}


def _safe_validate_or_empty(model_cls, text: str):  # noqa: ANN001
    try:
        return model_cls.model_validate(_safe_parse_json(text))
    except Exception:
        return model_cls()


def _content_metrics(target: int, o1: int, o2: int) -> tuple[float, float, bool]:
    c_value = max(0.0, min(1.0, target / 6.0))
    d_value = ((target - o1) + (target - o2)) / 2.0 / 6.0
    d_value = max(-1.0, min(1.0, d_value))
    meets = c_value >= 0.83 and d_value >= 0.35
    return c_value, d_value, meets


def build_deterministic_meta_review(
    *,
    content_review_text: str,
    linguistic_review_text: str,
    bias_review_text: str,
    meta_review_text: str,
) -> MetaEditorOutput:
    """Build final deterministic KEEP/REVISE/DISCARD decisions.

    Reviewer LLM outputs provide raw ratings. Numeric metrics and decisions are
    computed in code for deterministic behavior.
    """
    content = _safe_validate_or_empty(ContentReviewerOutput, content_review_text)
    linguistic = _safe_validate_or_empty(LinguisticReviewerOutput, linguistic_review_text)
    bias = _safe_validate_or_empty(BiasReviewerOutput, bias_review_text)
    meta = _safe_validate_or_empty(MetaEditorOutput, meta_review_text)

    ling_by_id = {i.item_number: i for i in linguistic.items}
    bias_by_id = {i.item_number: i for i in bias.items}
    meta_by_id = {i.item_number: i for i in meta.items}
    content_by_id = {i.item_number: i for i in content.items}

    item_ids = sorted(
        set(content_by_id.keys())
        | set(ling_by_id.keys())
        | set(bias_by_id.keys())
        | set(meta_by_id.keys())
    )

    decisions: list[MetaDecision] = []
    for item_id in item_ids:
        c = content_by_id.get(item_id)
        l = ling_by_id.get(item_id)
        b = bias_by_id.get(item_id)
        m = meta_by_id.get(item_id)

        target = c.target_rating if c else 3
        orbiting_1 = c.orbiting_1_rating if c else 3
        orbiting_2 = c.orbiting_2_rating if c else 3

        c_val, d_val, content_ok = _content_metrics(
            target=target,
            o1=orbiting_1,
            o2=orbiting_2,
        )
        bias_score = b.score if b else 3
        ling_min = min(
            [
                l.grammatical_accuracy if l else 3,
                l.ease_of_understanding if l else 3,
                l.negative_language_free if l else 3,
                l.clarity_directness if l else 3,
            ]
        )

        if content_ok and bias_score >= 4 and ling_min >= 4:
            decision = "KEEP"
        # Keep DISCARD conservative: use only for clearly problematic items.
        # Content mismatch alone should generally be revised, not discarded.
        elif bias_score <= 2 or ling_min <= 2:
            decision = "DISCARD"
        else:
            decision = "REVISE"

        reasons = [
            f"content(c={c_val:.2f}, d={d_val:.2f}, ok={content_ok})",
            f"ling_min={ling_min}",
            f"bias={bias_score}",
        ]
        if c and c.feedback:
            reasons.append(f"content_note={c.feedback}")
        elif c is None:
            reasons.append("content_note=missing_content_review")
        if l and l.feedback:
            reasons.append(f"ling_note={l.feedback}")
        if b and b.feedback:
            reasons.append(f"bias_note={b.feedback}")

        revised = None
        if decision == "REVISE" and m and m.revised_item_stem:
            revised = m.revised_item_stem

        decisions.append(
            MetaDecision(
                item_number=item_id,
                decision=decision,  # type: ignore[arg-type]
                reason="; ".join(reasons),
                revised_item_stem=revised,
            )
        )

    return MetaEditorOutput(
        items=decisions,
        overall_synthesis=(
            "Deterministic decisioning applied from reviewer raw ratings. "
            "Final decisions are code-computed."
        ),
    )
