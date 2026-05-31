"""enrich_related node: pull related papers from Semantic Scholar via MCP."""

from __future__ import annotations

from typing import Any

from packages.agent.clients import MCPClients
from packages.agent.state import AgentState

LIMIT = 8


async def run(state: AgentState, clients: MCPClients) -> dict[str, Any]:
    try:
        recs = await clients.call(
            "semantic_scholar",
            "recommendations",
            {"paper_id": f"arXiv:{state.arxiv_id}", "limit": LIMIT},
        )
        return {"related_papers": recs or []}
    except Exception as e:
        return {"errors": state.errors + [f"enrich_related: {type(e).__name__}: {e}"]}
