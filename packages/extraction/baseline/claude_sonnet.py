"""Claude Sonnet baseline: render first pages, ask the API for structured JSON.

No bbox grounding here — the API returns text only. Confidence is self-reported
per field (not from logprobs). This is the frontier-API baseline; expect strong
text fields and zero bboxes.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, cast

import pymupdf  # type: ignore[import-untyped]

from packages.extraction.schema import ExtractedField, Paper

MODEL = "claude-sonnet-4-6"
MAX_PAGES = 4
RENDER_DPI = 150
SELF_REPORTED_CONFIDENCE = 0.7

_PROMPT = """You are an information extractor for research papers. Read the page images
and return a JSON object with these fields. Use null or [] if a field is absent.

{
  "title": "string",
  "authors": ["string", ...],
  "abstract": "string",
  "methods": "string (1-3 sentence summary of the approach)",
  "datasets": ["string", ...],
  "tools_code": ["string", ...],
  "key_results": ["string", ...]
}

Return ONLY the JSON object. No prose, no markdown fences."""


def _render_pages(pdf_path: Path, max_pages: int) -> list[bytes]:
    out: list[bytes] = []
    with pymupdf.open(pdf_path) as doc:
        n = min(doc.page_count, max_pages)
        for i in range(n):
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


def _list(values: Any) -> list[ExtractedField[str]]:
    if not isinstance(values, list):
        return []
    return [
        ExtractedField(value=v.strip(), bbox=None, confidence=SELF_REPORTED_CONFIDENCE)
        for v in values
        if isinstance(v, str) and v.strip()
    ]


def extract(pdf_path: Path) -> Paper:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    # Import lazily so the module imports without anthropic installed.
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    images = _render_pages(pdf_path, MAX_PAGES)
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
        max_tokens=2048,
        messages=[{"role": "user", "content": cast(Any, content)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    data = _parse_json(text)
    return Paper(
        title=_scalar(data.get("title")),
        authors=_list(data.get("authors")),
        abstract=_scalar(data.get("abstract")),
        methods=_scalar(data.get("methods")),
        datasets=_list(data.get("datasets")),
        tools_code=_list(data.get("tools_code")),
        key_results=_list(data.get("key_results")),
    )
