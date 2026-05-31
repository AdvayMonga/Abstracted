"""write_note node: assemble Markdown + YAML frontmatter, send to obsidian_vault."""

from __future__ import annotations

import re
from typing import Any

from packages.agent.clients import MCPClients
from packages.agent.state import AgentState

_SLUG_BAD = re.compile(r"[^A-Za-z0-9 ._-]")


def _slug(title: str, arxiv_id: str) -> str:
    cleaned = _SLUG_BAD.sub("", title).strip()
    short = cleaned[:80].rstrip(" .-_")
    return f"{short or 'paper'} ({arxiv_id})"


def _render_body(state: AgentState) -> str:
    p = state.paper
    assert p is not None
    parts: list[str] = []
    if p.title:
        parts.append(f"# {p.title.value}\n")
    if p.abstract:
        parts.append(f"## Abstract\n\n{p.abstract.value}\n")
    if p.methods:
        parts.append(f"## Methods\n\n{p.methods.value}\n")
    if p.key_results:
        bullets = "\n".join(f"- {kr.value}" for kr in p.key_results)
        parts.append(f"## Key results\n\n{bullets}\n")
    if state.code_urls:
        bullets = "\n".join(f"- {u}" for u in state.code_urls)
        parts.append(f"## Code\n\n{bullets}\n")
    if state.related_papers:
        bullets = "\n".join(
            f"- {r.get('title', '?')} ({r.get('year', '?')})" for r in state.related_papers
        )
        parts.append(f"## Related work\n\n{bullets}\n")
    return "\n".join(parts)


def _render_frontmatter(state: AgentState) -> dict[str, Any]:
    p = state.paper
    assert p is not None
    fm: dict[str, Any] = {"arxiv_id": state.arxiv_id}
    if p.title:
        fm["title"] = p.title.value
    if p.authors:
        fm["authors"] = [a.value for a in p.authors]
    if state.code_urls:
        fm["code"] = state.code_urls
    return fm


async def run(state: AgentState, clients: MCPClients) -> dict[str, Any]:
    if state.paper is None or state.paper.title is None:
        return {"errors": state.errors + ["write_note: no paper/title in state"]}
    slug = _slug(state.paper.title.value, state.arxiv_id)
    try:
        path = await clients.call(
            "obsidian_vault",
            "write_note",
            {
                "slug": slug,
                "body": _render_body(state),
                "frontmatter": _render_frontmatter(state),
            },
        )
        return {"note_path": str(path)}
    except Exception as e:
        return {"errors": state.errors + [f"write_note: {type(e).__name__}: {e}"]}
