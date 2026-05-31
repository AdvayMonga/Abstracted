"""Claude Sonnet baseline: render selected pages, ask the API for structured JSON.

No bbox grounding here — the API returns text only. Confidence is self-reported
per field (not from logprobs). This is the frontier-API baseline.

Page selection: first FRONT_PAGES (title/abstract/intro/methods) plus the last
BACK_PAGES (references). Keeps token cost down vs. rendering everything.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, cast

import pymupdf  # type: ignore[import-untyped]

from packages.extraction.schema import Citation, ExtractedField, Paper

MODEL = "claude-sonnet-4-6"
FRONT_PAGES = 6
BACK_PAGES = 3
RENDER_DPI = 144
MAX_TOKENS = 8192
SELF_REPORTED_CONFIDENCE = 0.7

_PROMPT = """You are an extractor for research papers. Read the page images and return
a JSON object with the fields below. Use null or [] if a field is absent.

Fields:
- title (string)
- authors (list of strings)
- abstract (string)
- methods (string) — copy the methods/approach section verbatim, joining
  paragraphs with single spaces. Do not paraphrase or summarize. If the section
  is split across pages we did not show you, return what is visible. If absent,
  null.
- datasets (list of strings) — dataset names referenced in the paper.
- tools_code (list of strings) — full https://github.com/... URLs only.
- key_results (list of strings) — short sentences stating numeric results, SOTA
  claims, or improvements over baselines.
- citations (list of objects with keys: raw, title, year) — every reference in
  the references section. "raw" is the full printed string. year is an int or
  null. title may be null.

Return ONLY the JSON object. No prose, no markdown fences."""


def _select_pages(page_count: int) -> list[int]:
    """First FRONT_PAGES + last BACK_PAGES, deduped, in order."""
    front = list(range(min(FRONT_PAGES, page_count)))
    back = list(range(max(0, page_count - BACK_PAGES), page_count))
    seen: set[int] = set()
    ordered: list[int] = []
    for p in front + back:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _render(pdf_path: Path) -> list[bytes]:
    out: list[bytes] = []
    with pymupdf.open(pdf_path) as doc:
        for i in _select_pages(doc.page_count):
            pix = doc[i].get_pixmap(dpi=RENDER_DPI)
            out.append(cast(bytes, pix.tobytes("png")))
    return out


def _parse_json(raw: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _scalar(value: Any) -> ExtractedField[str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return ExtractedField(value=value.strip(), bbox=None, confidence=SELF_REPORTED_CONFIDENCE)


def _list_str(values: Any) -> list[ExtractedField[str]]:
    if not isinstance(values, list):
        return []
    return [
        ExtractedField(value=v.strip(), bbox=None, confidence=SELF_REPORTED_CONFIDENCE)
        for v in values
        if isinstance(v, str) and v.strip()
    ]


def _citations(values: Any) -> list[ExtractedField[Citation]]:
    if not isinstance(values, list):
        return []
    out: list[ExtractedField[Citation]] = []
    for v in values:
        if not isinstance(v, dict):
            continue
        raw = v.get("raw")
        if not isinstance(raw, str) or not raw.strip():
            continue
        year = v.get("year")
        if not isinstance(year, int):
            year = None
        title = v.get("title") if isinstance(v.get("title"), str) else None
        cit = Citation(raw=raw.strip(), title=title, year=year)
        out.append(ExtractedField(value=cit, bbox=None, confidence=SELF_REPORTED_CONFIDENCE))
    return out


def extract(pdf_path: Path) -> Paper:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    images = _render(pdf_path)
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(img).decode(),
            },
        }
        for img in images
    ]
    content.append({"type": "text", "text": _PROMPT})

    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": cast(Any, content)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    data = _parse_json(text)
    if not data:
        print(
            f"[claude_sonnet] empty/unparseable response for {pdf_path.name}, "
            f"stop_reason={msg.stop_reason}, text head: {text[:160]!r}"
        )
    return Paper(
        title=_scalar(data.get("title")),
        authors=_list_str(data.get("authors")),
        abstract=_scalar(data.get("abstract")),
        methods=_scalar(data.get("methods")),
        datasets=_list_str(data.get("datasets")),
        tools_code=_list_str(data.get("tools_code")),
        key_results=_list_str(data.get("key_results")),
        citations=_citations(data.get("citations")),
    )
