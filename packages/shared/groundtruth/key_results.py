"""Heuristic key_results: sentences from results/conclusion sections with numbers or
explicit improvement language."""

from __future__ import annotations

import re

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# A sentence qualifies if it contains a numeric metric, a benchmark name,
# or an explicit improvement verb.
_NUMERIC = re.compile(
    r"(\d+(?:\.\d+)?\s*%"
    r"|\+\s*\d+(?:\.\d+)?"
    r"|\b\d+(?:\.\d+)?\s*(?:points?|pp|x)\b"
    r"|\bF1\b|\bBLEU\b|\bROUGE\b|\bAUC\b|\bmAP\b|\bPSNR\b|\bIoU\b"
    r"|\baccuracy\b|\bprecision\b|\brecall\b"
    r"|\bstate[- ]of[- ]the[- ]art\b|\bSOTA\b"
    r"|\b(?:outperform|outperforms|improves?\s+(?:on|over|by)|achieves?|surpass|reduces?\s+by|exceeds?)\b"
    r")",
    re.IGNORECASE,
)


def extract(
    results_text: str | None,
    max_n: int = 5,
    fallback_text: str | None = None,
) -> list[str]:
    """Extract key result sentences. Falls back to `fallback_text` (typically the
    conclusion/discussion section) if the results section yields none."""
    primary = _scan(results_text, max_n)
    if primary or not fallback_text:
        return primary
    return _scan(fallback_text, max_n)


def _scan(text: str | None, max_n: int) -> list[str]:
    if not text:
        return []
    sentences = _SENTENCE.split(text)
    hits: list[str] = []
    for raw in sentences:
        s = raw.strip()
        if 40 <= len(s) <= 400 and _NUMERIC.search(s):
            hits.append(s)
        if len(hits) >= max_n:
            break
    return hits
