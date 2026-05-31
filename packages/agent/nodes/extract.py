"""extract node: PDF → Paper via Claude Sonnet.

Not MCP — extraction is layer 1 of the system, separate from the tool layer.
The node calls the local baseline directly.
"""

from __future__ import annotations

from typing import Any

from packages.agent.state import AgentState
from packages.extraction.baseline.claude_sonnet import extract as claude_extract


async def run(state: AgentState, clients: Any) -> dict[str, Any]:  # noqa: ARG001
    try:
        paper = claude_extract(state.pdf_path)
        return {"paper": paper}
    except Exception as e:
        return {"errors": state.errors + [f"extract: {type(e).__name__}: {e}"]}
