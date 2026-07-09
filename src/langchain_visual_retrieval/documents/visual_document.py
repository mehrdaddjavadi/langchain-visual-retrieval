"""Visual retrieval result model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VisualDocument(BaseModel):
    """A lightweight internal representation for visual retrieval results."""

    id: str | None = Field(default=None)
    source: str | None = Field(default=None)
    page: int | None = Field(default=None)
    image_path: str | None = Field(default=None)
    score: float | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
