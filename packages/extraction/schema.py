"""Extraction schema. Every field carries value, bbox, confidence."""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class BBox(BaseModel):
    """Page-indexed bounding box. Coords in PDF points, origin top-left."""

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=0)
    x0: float
    y0: float
    x1: float
    y1: float


class ExtractedField(BaseModel, Generic[T]):
    """A single extracted value with provenance."""

    value: T
    bbox: BBox | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class Citation(BaseModel):
    """A single reference entry. Sub-fields are not individually bbox'd."""

    raw: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None


class Paper(BaseModel):
    """Top-level extraction target. Mirrors Nanonets OCR-3 output shape."""

    title: ExtractedField[str] | None = None
    authors: list[ExtractedField[str]] = Field(default_factory=list)
    abstract: ExtractedField[str] | None = None
    methods: ExtractedField[str] | None = None
    datasets: list[ExtractedField[str]] = Field(default_factory=list)
    tools_code: list[ExtractedField[str]] = Field(default_factory=list)
    key_results: list[ExtractedField[str]] = Field(default_factory=list)
    citations: list[ExtractedField[Citation]] = Field(default_factory=list)
