"""Field-level and list-level similarity metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from packages.extraction.schema import Citation, ExtractedField

CORRECT_THRESHOLD = 0.85
LIST_MATCH_THRESHOLD = 0.7

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())


def scalar_similarity(pred: str | None, gold: str | None) -> float:
    if pred is None and gold is None:
        return 1.0
    if pred is None or gold is None:
        return 0.0
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return 1.0
    return fuzz.ratio(p, g) / 100.0


def _item_text(item: ExtractedField[str] | ExtractedField[Citation]) -> str:
    v = item.value
    if isinstance(v, Citation):
        return v.title or v.raw
    return v


@dataclass(frozen=True)
class ListScore:
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_gold: int
    n_matched: int


def list_score(
    pred: list[ExtractedField[str]] | list[ExtractedField[Citation]],
    gold: list[ExtractedField[str]] | list[ExtractedField[Citation]],
    threshold: float = LIST_MATCH_THRESHOLD,
) -> ListScore:
    if not pred and not gold:
        return ListScore(1.0, 1.0, 1.0, 0, 0, 0)
    if not pred or not gold:
        return ListScore(0.0, 0.0, 0.0, len(pred), len(gold), 0)

    p_text = [_item_text(x) for x in pred]
    g_text = [_item_text(x) for x in gold]
    sim = np.zeros((len(p_text), len(g_text)))
    for i, p in enumerate(p_text):
        for j, g in enumerate(g_text):
            sim[i, j] = scalar_similarity(p, g)

    # Hungarian maximizes assignment over a cost matrix; we minimize -sim.
    row_idx, col_idx = linear_sum_assignment(-sim)
    matched = int(
        sum(1 for r, c in zip(row_idx, col_idx, strict=True) if sim[r, c] >= threshold)
    )
    precision = matched / len(pred)
    recall = matched / len(gold)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return ListScore(precision, recall, f1, len(pred), len(gold), matched)
