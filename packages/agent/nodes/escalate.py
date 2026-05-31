"""escalate node: send the paper to the review queue and stop the pipeline."""

from __future__ import annotations

from typing import Any

from packages.agent.clients import MCPClients
from packages.agent.state import AgentState


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
        return {"review_item_id": int(item_id)}
    except Exception as e:
        return {"errors": state.errors + [f"escalate: {type(e).__name__}: {e}"]}
