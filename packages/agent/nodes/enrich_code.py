"""enrich_code node: find GitHub repos for the paper via MCP.

Tries the PDF scanner first (covers recent papers PwC hasn't indexed) and
falls back to Papers With Code. Results merged and deduped.
"""

from __future__ import annotations

from typing import Any

from packages.agent.clients import MCPClients
from packages.agent.state import AgentState


async def run(state: AgentState, clients: MCPClients) -> dict[str, Any]:
    urls: list[str] = []
    errors = list(state.errors)
    try:
        scanned = await clients.call(
            "github_lookup", "scan_pdf", {"pdf_path": str(state.pdf_path)}
        )
        if isinstance(scanned, list):
            urls.extend(scanned)
    except Exception as e:
        errors.append(f"enrich_code.scan_pdf: {type(e).__name__}: {e}")

    try:
        pwc = await clients.call("github_lookup", "pwc_lookup", {"arxiv_id": state.arxiv_id})
        if isinstance(pwc, str):
            urls.append(pwc)
    except Exception as e:
        errors.append(f"enrich_code.pwc_lookup: {type(e).__name__}: {e}")

    deduped: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return {"code_urls": deduped, "errors": errors}
