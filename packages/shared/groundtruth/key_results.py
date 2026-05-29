"""Heuristic key_results: sentences in the Results section with numeric improvements."""

from __future__ import annotations

import re

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_NUMERIC = re.compile(
    r"(\d+(?:\.\d+)?\s*%"
    r"|\+\s*\d+(?:\.\d+)?"
    r"|\bF1\b|\bBLEU\b|\bROUGE\b|\baccuracy\b|\bAUC\b"
    r"|\bstate[- ]of[- ]the[- ]art\b|\bSOTA\b)",
    re.IGNORECASE,
)


def extract(results_text: str | None, max_n: int = 5) -> list[str]:
    if not results_text:
        return []
    sentences = _SENTENCE.split(results_text)
    hits: list[str] = []
    for s in sentences:
        s = s.strip()
        if 40 <= len(s) <= 400 and _NUMERIC.search(s):
            hits.append(s)
        if len(hits) >= max_n:
            break
    return hits
