"""Score a predicted Paper against a ground-truth Paper, then aggregate across a dataset."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from packages.extraction.eval.metrics import (
    CORRECT_THRESHOLD,
    ListScore,
    list_score,
    scalar_similarity,
)
from packages.extraction.schema import ExtractedField, Paper

SCALAR_FIELDS = ("title", "abstract", "methods")
LIST_FIELDS = ("authors", "datasets", "tools_code", "key_results", "citations")


def _scalar_value(f: ExtractedField[str] | None) -> str | None:
    return f.value if f else None


@dataclass
class DocScore:
    doc_id: str
    scalar: dict[str, float] = field(default_factory=dict)
    lists: dict[str, ListScore] = field(default_factory=dict)

    @property
    def all_fields_correct(self) -> bool:
        scalars_ok = all(s >= CORRECT_THRESHOLD for s in self.scalar.values())
        lists_ok = all(ls.f1 >= CORRECT_THRESHOLD for ls in self.lists.values())
        return scalars_ok and lists_ok


def score_document(doc_id: str, pred: Paper, gold: Paper) -> DocScore:
    out = DocScore(doc_id=doc_id)
    for name in SCALAR_FIELDS:
        out.scalar[name] = scalar_similarity(
            _scalar_value(getattr(pred, name)),
            _scalar_value(getattr(gold, name)),
        )
    for name in LIST_FIELDS:
        out.lists[name] = list_score(getattr(pred, name), getattr(gold, name))
    return out


@dataclass
class DatasetReport:
    per_doc: list[DocScore]
    scalar_mean: dict[str, float]
    list_f1_mean: dict[str, float]
    pct_fully_correct: float

    @property
    def macro_f1(self) -> float:
        vals = list(self.scalar_mean.values()) + list(self.list_f1_mean.values())
        return sum(vals) / len(vals) if vals else 0.0


def aggregate(scores: Iterable[DocScore]) -> DatasetReport:
    scores = list(scores)
    if not scores:
        return DatasetReport([], {}, {}, 0.0)
    scalar_mean = {
        name: sum(s.scalar[name] for s in scores) / len(scores) for name in SCALAR_FIELDS
    }
    list_f1_mean = {
        name: sum(s.lists[name].f1 for s in scores) / len(scores) for name in LIST_FIELDS
    }
    pct = sum(1 for s in scores if s.all_fields_correct) / len(scores)
    return DatasetReport(scores, scalar_mean, list_f1_mean, pct)
