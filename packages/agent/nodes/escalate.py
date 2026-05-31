"""escalate node: enqueue to review_queue, then interrupt for a human decision.

The runtime persists state via the checkpointer when interrupt() raises, and
the node only "returns" once something resumes the thread with Command(resume=<decision>).
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from packages.agent.clients import MCPClients
from packages.agent.state import AgentState

VALID_DECISIONS = {"approve", "reject", "defer"}


def _payload(state: AgentState) -> dict[str, Any]:
    p = state.paper
    if p is None:
        return {"pdf_path": str(state.pdf_path)}
    return {
        "pdf_path": str(state.pdf_path),
        "title": p.title.value if p.title else None,
        "abstract_present": p.abstract is not None,
        "authors": [a.value for a in p.authors],
    }


async def run(state: AgentState, clients: MCPClients) -> dict[str, Any]:
    if state.review_item_id is None:
        try:
            item_id = await clients.call(
                "review_queue",
                "enqueue",
                {
                    "paper_id": state.arxiv_id,
                    "reason": ",".join(state.review_reasons) or "unspecified",
                    "payload": _payload(state),
                },
            )
            item_id = int(item_id)
        except Exception as e:
            return {"errors": state.errors + [f"escalate: {type(e).__name__}: {e}"]}
    else:
        item_id = state.review_item_id

    decision = interrupt(
        {
            "kind": "review_needed",
            "review_item_id": item_id,
            "reasons": state.review_reasons,
            "arxiv_id": state.arxiv_id,
        }
    )
    if decision not in VALID_DECISIONS:
        return {
            "review_item_id": item_id,
            "errors": state.errors + [f"escalate: invalid decision {decision!r}"],
        }

    # Reflect the human's decision in the review_queue too.
    try:
        status = "approved" if decision == "approve" else (
            "rejected" if decision == "reject" else "deferred"
        )
        await clients.call(
            "review_queue", "resolve", {"item_id": item_id, "status": status}
        )
    except Exception as e:
        return {
            "review_item_id": item_id,
            "review_decision": decision,
            "errors": state.errors + [f"escalate.resolve: {type(e).__name__}: {e}"],
        }

    return {"review_item_id": item_id, "review_decision": decision}


def after_escalate(state: AgentState) -> str:
    """Conditional-edge router. approve → continue pipeline, else end."""
    return "enrich_related" if state.review_decision == "approve" else "end"
