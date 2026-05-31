"""Build the LangGraph state machine.

  extract -> triage -> [escalate -> END]
                    \\-> enrich_related -> enrich_code -> write_note -> END

Conditional edge after triage. Postgres checkpointer + interrupt() come in
slice 3.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from packages.agent.clients import MCPClients
from packages.agent.nodes import (
    enrich_code,
    enrich_related,
    escalate,
    extract,
    triage,
    write_note,
)
from packages.agent.state import AgentState


def build_graph(clients: MCPClients) -> Any:
    """Return a compiled LangGraph runnable bound to the given MCP session pool."""

    async def _extract(state: AgentState) -> dict[str, Any]:
        return await extract.run(state, clients)

    async def _triage(state: AgentState) -> dict[str, Any]:
        return await triage.run(state, clients)

    async def _escalate(state: AgentState) -> dict[str, Any]:
        return await escalate.run(state, clients)

    async def _related(state: AgentState) -> dict[str, Any]:
        return await enrich_related.run(state, clients)

    async def _code(state: AgentState) -> dict[str, Any]:
        return await enrich_code.run(state, clients)

    async def _note(state: AgentState) -> dict[str, Any]:
        return await write_note.run(state, clients)

    g: StateGraph = StateGraph(AgentState)
    g.add_node("extract", _extract)
    g.add_node("triage", _triage)
    g.add_node("escalate", _escalate)
    g.add_node("enrich_related", _related)
    g.add_node("enrich_code", _code)
    g.add_node("write_note", _note)

    g.add_edge(START, "extract")
    g.add_edge("extract", "triage")
    g.add_conditional_edges(
        "triage",
        triage.needs_review,
        {"escalate": "escalate", "enrich_related": "enrich_related"},
    )
    g.add_edge("escalate", END)
    g.add_edge("enrich_related", "enrich_code")
    g.add_edge("enrich_code", "write_note")
    g.add_edge("write_note", END)

    return g.compile()
