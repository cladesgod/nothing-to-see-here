"""Structured JSON output schemas for all LLM agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class WebSurferOutput(BaseModel):
    research_summary: str = Field(..., min_length=20)
    key_points: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class ItemOut(BaseModel):
    item_number: int
    stem: str = Field(..., min_length=5)
    rationale: str = Field(..., min_length=5)


class ItemWriterOutput(BaseModel):
    items: list[ItemOut] = Field(..., min_length=1)
    response_scale: str = "1 (Strongly Disagree) to 7 (Strongly Agree)"


class ContentReviewItem(BaseModel):
    item_number: int
    target_rating: int
    orbiting_1_rating: int
    orbiting_2_rating: int
    feedback: str = ""


class ContentReviewerOutput(BaseModel):
    items: list[ContentReviewItem] = Field(default_factory=list)
    overall_summary: str = ""


class LinguisticReviewItem(BaseModel):
    item_number: int
    grammatical_accuracy: int
    ease_of_understanding: int
    negative_language_free: int
    clarity_directness: int
    feedback: str = ""


class LinguisticReviewerOutput(BaseModel):
    items: list[LinguisticReviewItem] = Field(default_factory=list)
    overall_summary: str = ""


class BiasReviewItem(BaseModel):
    item_number: int
    score: int
    feedback: str = ""


class BiasReviewerOutput(BaseModel):
    items: list[BiasReviewItem] = Field(default_factory=list)
    overall_summary: str = ""


class MetaDecision(BaseModel):
    item_number: int
    decision: Literal["KEEP", "REVISE", "DISCARD"]
    reason: str = ""
    revised_item_stem: str | None = None


class MetaEditorOutput(BaseModel):
    items: list[MetaDecision] = Field(default_factory=list)
    overall_synthesis: str = ""


class LewModOutput(BaseModel):
    decision: Literal["APPROVE", "REVISE"]
    feedback: str
    keep: list[int] = Field(default_factory=list)
    revise: list[int] = Field(default_factory=list)
    discard: list[int] = Field(default_factory=list)
