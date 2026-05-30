"""Locate a text span's bbox on a PDF page via word-level substring search.

Handles hyphenated line breaks ("approa-\nch" → "approach") and returns a
union bbox over the full matched span (not just the anchor prefix).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

import pymupdf  # type: ignore[import-untyped]

from packages.extraction.schema import BBox

_WS = re.compile(r"\s+")
_ANCHOR_TOKENS = 5
_LOOKAHEAD = 4
# Punctuation that PyMuPDF doesn't always treat as a word boundary but should be.
_SPLIT_CHARS = re.compile(r"[—–:;]+")


def _norm(s: str) -> str:
    # Replace em/en dash, colon, semicolon with spaces so they tokenize the same way
    # whether they appear in the query or in PDF text (PyMuPDF doesn't split on them).
    s = _SPLIT_CHARS.sub(" ", s)
    return _WS.sub(" ", s.strip().lower())


@dataclass(frozen=True)
class _Word:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def _split_word(w: _Word) -> list[_Word]:
    """Split a word on em-dash/en-dash/colon/semicolon. Each sub-word keeps the bbox."""
    parts = [p for p in _SPLIT_CHARS.split(w.text) if p]
    if len(parts) <= 1:
        return [w]
    return [_Word(w.page, w.x0, w.y0, w.x1, w.y1, p) for p in parts]


def _collect_words(doc: pymupdf.Document, max_pages: int = 2) -> list[_Word]:
    out: list[_Word] = []
    n = min(doc.page_count, max_pages)
    for i in range(n):
        words = cast(list[Any], doc[i].get_text("words"))
        for w in words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            base = _Word(i, x0, y0, x1, y1, text)
            out.extend(_split_word(base))
    return out


def _merge_hyphens(words: list[_Word]) -> list[_Word]:
    """Join words ending in '-' with the next word on the following line."""
    merged: list[_Word] = []
    i = 0
    while i < len(words):
        w = words[i]
        if (
            w.text.endswith("-")
            and i + 1 < len(words)
            and words[i + 1].page == w.page
            and words[i + 1].y0 > w.y0
        ):
            nxt = words[i + 1]
            joined_text = w.text[:-1] + nxt.text
            merged.append(
                _Word(
                    page=w.page,
                    x0=min(w.x0, nxt.x0),
                    y0=w.y0,
                    x1=max(w.x1, nxt.x1),
                    y1=nxt.y1,
                    text=joined_text,
                )
            )
            i += 2
        else:
            merged.append(w)
            i += 1
    return merged


def _union(run: list[_Word]) -> BBox:
    page = run[0].page
    same = [w for w in run if w.page == page]
    return BBox(
        page=page,
        x0=min(w.x0 for w in same),
        y0=min(w.y0 for w in same),
        x1=max(w.x1 for w in same),
        y1=max(w.y1 for w in same),
    )


def locate(doc: pymupdf.Document, query: str, max_pages: int = 2) -> BBox | None:
    """Find the bbox of `query` in the first `max_pages` pages.

    Returns the union bbox of the matched word run. Returns None if no
    anchor (first ~5 tokens) can be matched.
    """
    target = _norm(query)
    if not target:
        return None
    tokens = target.split(" ")
    if not tokens:
        return None

    words = _merge_hyphens(_collect_words(doc, max_pages=max_pages))
    if not words:
        return None
    norm_words = [_norm(w.text) for w in words]

    # Exact full-phrase match first.
    n = len(tokens)
    if n <= len(words):
        for i in range(len(words) - n + 1):
            if norm_words[i : i + n] == tokens and words[i].page == words[i + n - 1].page:
                return _union(words[i : i + n])

    # Anchor on the first ANCHOR_TOKENS, then walk forward consuming subsequent
    # query tokens with a small lookahead window (tolerates dropped/merged tokens).
    anchor = tokens[: min(_ANCHOR_TOKENS, n)]
    a = len(anchor)
    for i in range(len(words) - a + 1):
        if norm_words[i : i + a] != anchor:
            continue
        # Found anchor at i. Greedy extend.
        run_start = i
        run_end = i + a  # exclusive
        cursor = run_end
        page = words[i].page
        for tok in tokens[a:]:
            found = False
            for j in range(cursor, min(cursor + _LOOKAHEAD, len(words))):
                if words[j].page != page:
                    break
                if norm_words[j] == tok:
                    run_end = j + 1
                    cursor = j + 1
                    found = True
                    break
            if not found:
                cursor += 1  # advance past one PDF word and keep trying
                if cursor > len(words) or (cursor < len(words) and words[cursor].page != page):
                    break
        return _union(words[run_start:run_end])

    return None
