"""Trivial baseline: PyMuPDF text + regex. Floor for the eval harness."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pymupdf  # type: ignore[import-untyped]

from packages.extraction.schema import BBox, ExtractedField, Paper

BASELINE_CONFIDENCE = 0.3

_SECTION_HEADERS = re.compile(
    r"^\s*(?:\d+\.?\s*)?(introduction|background|related work|methods?|methodology|"
    r"experiments?|results?|discussion|conclusions?|references?)\b",
    re.IGNORECASE | re.MULTILINE,
)
_AUTHOR_SPLIT = re.compile(r",\s*|\s+and\s+", re.IGNORECASE)


def _first_page_blocks(doc: pymupdf.Document) -> list[dict]:
    """Return text blocks on page 1 with font-size info."""
    if doc.page_count == 0:
        return []
    page = doc[0]
    data = cast(dict[str, Any], page.get_text("dict"))
    blocks = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            max_size = max(s["size"] for s in spans)
            bbox = line["bbox"]
            blocks.append({"text": text, "size": max_size, "bbox": bbox})
    return blocks


def _extract_title(blocks: list[dict]) -> ExtractedField[str] | None:
    if not blocks:
        return None
    # Largest font line in the top half of the page.
    top = [b for b in blocks if b["bbox"][1] < 400] or blocks
    best = max(top, key=lambda b: b["size"])
    x0, y0, x1, y1 = best["bbox"]
    return ExtractedField(
        value=best["text"],
        bbox=BBox(page=0, x0=x0, y0=y0, x1=x1, y1=y1),
        confidence=BASELINE_CONFIDENCE,
    )


def _extract_authors(
    blocks: list[dict], title_field: ExtractedField[str] | None
) -> list[ExtractedField[str]]:
    if not blocks or title_field is None or title_field.bbox is None:
        return []
    # First line immediately under the title that isn't another large heading.
    title_y = title_field.bbox.y1
    candidates = [b for b in blocks if b["bbox"][1] > title_y and b["bbox"][1] < title_y + 120]
    if not candidates:
        return []
    candidates.sort(key=lambda b: b["bbox"][1])
    line = candidates[0]
    names = [n.strip() for n in _AUTHOR_SPLIT.split(line["text"]) if n.strip()]
    # Reject obviously-not-authors lines (too long, contain numbers heavily).
    if not names or any(len(n) > 60 for n in names):
        return []
    x0, y0, x1, y1 = line["bbox"]
    bbox = BBox(page=0, x0=x0, y0=y0, x1=x1, y1=y1)
    return [
        ExtractedField(value=n, bbox=bbox, confidence=BASELINE_CONFIDENCE) for n in names
    ]


def _extract_abstract(doc: pymupdf.Document) -> ExtractedField[str] | None:
    if doc.page_count == 0:
        return None
    text = cast(str, doc[0].get_text("text"))
    m = re.search(
        r"\babstract\b[\s:.\-]*\n(.+?)(?=\n\s*(?:1\.\s|\d+\s|introduction\b|keywords?\b))",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        m = re.search(
            r"\babstract\b[\s:.\-]*\n(.{50,3000}?)\n\s*\n",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    if not m:
        return None
    body = re.sub(r"\s+", " ", m.group(1)).strip()
    if len(body) < 50:
        return None
    return ExtractedField(value=body, bbox=None, confidence=BASELINE_CONFIDENCE)


def extract(pdf_path: Path) -> Paper:
    with pymupdf.open(pdf_path) as doc:
        blocks = _first_page_blocks(doc)
        title = _extract_title(blocks)
        authors = _extract_authors(blocks, title)
        abstract = _extract_abstract(doc)
    return Paper(title=title, authors=authors, abstract=abstract)
