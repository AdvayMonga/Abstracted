"""Agent state. One Pydantic model the whole graph reads and writes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from packages.extraction.schema import Paper


class AgentState(BaseModel):
    arxiv_id: str
    pdf_path: Path

    paper: Paper | None = None
    related_papers: list[dict[str, Any]] = Field(default_factory=list)
    code_urls: list[str] = Field(default_factory=list)
    note_path: str | None = None

    errors: list[str] = Field(default_factory=list)
