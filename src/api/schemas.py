"""Request/response Pydantic models for the API layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ConstructDefinition(BaseModel):
    """Inline construct definition for custom constructs."""

    name: str = Field(..., min_length=1, max_length=200)
    definition: str = Field(..., min_length=10, max_length=5000)
    dimensions: list[DimensionInput] = Field(..., min_length=1, max_length=20)


class DimensionInput(BaseModel):
    """A single dimension with orbiting dimensions."""

    name: str = Field(..., min_length=1, max_length=200)
    definition: str = Field(..., min_length=5, max_length=2000)
    orbiting: list[str] = Field(default_factory=list, max_length=10)


# Fix forward reference
ConstructDefinition.model_rebuild()


class RunCreateRequest(BaseModel):
    """Request body for creating a new pipeline run."""

    model_config = ConfigDict(populate_by_name=True)

    preset: str | None = Field(
        default=None,
        description="Built-in construct preset name (e.g., 'aaaw').",
    )
    construct_definition: ConstructDefinition | None = Field(
        default=None,
        alias="construct",
        description="Custom construct definition (mutually exclusive with preset).",
    )
    lewmod: bool = Field(
        default=False,
        description="Use LewMod (automated LLM expert) instead of human feedback.",
    )
    max_revisions: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum revision rounds. None = use default from config.",
    )


class FeedbackRequest(BaseModel):
    """Request body for submitting human feedback on a paused run."""

    approve: bool = Field(
        default=False,
        description="If True, approve all items and finish the run.",
    )
    item_decisions: dict[str, str] = Field(
        default_factory=dict,
        description="Per-item decisions: {'1': 'KEEP', '2': 'REVISE', ...}",
    )
    global_note: str = Field(
        default="",
        max_length=2000,
        description="Optional free-text note for the revision round.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class RunStatusResponse(BaseModel):
    """Response for run status queries."""

    run_id: str
    status: str = Field(description="running | waiting_feedback | done | failed | cancelled")
    phase: str | None = Field(default=None, description="Current pipeline phase")
    construct_name: str
    mode: str = Field(description="human | lewmod")
    revision_count: int = 0
    max_revisions: int = 5
    items_text: str | None = None
    review_text: str | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None


class RunListResponse(BaseModel):
    """Paginated list of runs."""

    runs: list[RunStatusResponse]
    total: int
    page: int
    page_size: int


class RunCreatedResponse(BaseModel):
    """Response after successfully submitting a new run."""

    run_id: str
    status: str = "queued"
    message: str = "Run submitted successfully."


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    queue_depth: int = 0
    active_workers: int = 0
    max_workers: int = 10
    db_connected: bool = True
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
