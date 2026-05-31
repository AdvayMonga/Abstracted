"""triage node: decide whether the extraction is good enough to publish.

Populates `state.review_reasons` with one string per failed check. The
conditional edge in graph.py reads that list to route. Doesn't touch other
fields — keeps escalation logic in one place.
"""

from __future__ import annotations

from typing import Any

from packages.agent.state import AgentState
from packages.extraction.schema import ExtractedField

THRESHOLD = 0.5
REQUIRED_SCALARS: tuple[str, ...] = ("title", "abstract")


def _under_threshold(f: ExtractedField | None) -> bool:
    return f is not None and f.confidence < THRESHOLD


async def run(state: AgentState, clients: Any) -> dict[str, Any]:  # noqa: ARG001
    reasons: list[str] = []
    paper = state.paper
    if paper is None:
        reasons.append("extract_failed")
        return {"review_reasons": reasons}

    for name in REQUIRED_SCALARS:
        field = getattr(paper, name)
        if field is None:
            reasons.append(f"missing:{name}")
        elif _under_threshold(field):
            reasons.append(f"low_conf:{name}")

    if not paper.authors:
        reasons.append("missing:authors")

    return {"review_reasons": reasons}


def needs_review(state: AgentState) -> str:
    """Conditional-edge router. Returns the next node name."""
    return "escalate" if state.review_reasons else "enrich_related"
