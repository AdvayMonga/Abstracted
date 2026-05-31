"""Graph-level test with a fake MCPClients. No subprocesses, no API calls."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pymupdf  # type: ignore[import-untyped]
import pytest

from packages.agent.graph import build_graph
from packages.agent.state import AgentState
from packages.extraction.schema import ExtractedField, Paper


class FakeClients:
    """Drop-in for MCPClients with canned tool responses."""

    def __init__(self, responses: dict[tuple[str, str], Any]):
        self._responses = responses
        self.calls: list[tuple[str, str, dict]] = []

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        self.calls.append((server, tool, args))
        return self._responses.get((server, tool))


@pytest.fixture
def pdf_path(tmp_path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Fake Paper for Graph Test", fontsize=18)
    out = tmp_path / "2506.11419.pdf"
    doc.save(out)
    doc.close()
    return out


def _good_paper() -> Paper:
    return Paper(
        title=ExtractedField(value="Fake Paper", confidence=0.9),
        authors=[ExtractedField(value="A. Author", confidence=0.9)],
        abstract=ExtractedField(value="Test abstract.", confidence=0.9),
    )


def test_graph_happy_path(pdf_path, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACTED_VAULT_DIR", str(tmp_path / "vault"))
    import packages.agent.nodes.extract as extract_node

    monkeypatch.setattr(extract_node, "claude_extract", lambda _p: _good_paper())

    note_path = str(tmp_path / "vault" / "Fake Paper (2506.11419).md")
    clients = FakeClients(
        {
            ("semantic_scholar", "recommendations"): [
                {"title": "Related One", "year": 2024},
                {"title": "Related Two", "year": 2025},
            ],
            ("github_lookup", "scan_pdf"): ["https://github.com/foo/bar"],
            ("github_lookup", "pwc_lookup"): "https://github.com/foo/bar",
            ("obsidian_vault", "write_note"): note_path,
        }
    )

    initial = AgentState(arxiv_id="2506.11419", pdf_path=pdf_path)
    graph = build_graph(clients)  # type: ignore[arg-type]
    final = AgentState.model_validate(asyncio.run(graph.ainvoke(initial)))

    assert final.paper is not None
    assert final.note_path == note_path
    assert final.review_reasons == []
    assert final.review_item_id is None
    assert final.errors == []
    called = {(s, t) for s, t, _ in clients.calls}
    assert ("semantic_scholar", "recommendations") in called
    assert ("obsidian_vault", "write_note") in called
    # escalate path was NOT taken
    assert ("review_queue", "enqueue") not in called


def test_extract_failure_routes_to_escalate(pdf_path, monkeypatch) -> None:
    import packages.agent.nodes.extract as extract_node

    def boom(_path):
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(extract_node, "claude_extract", boom)
    clients = FakeClients({("review_queue", "enqueue"): 42})
    initial = AgentState(arxiv_id="2506.11419", pdf_path=pdf_path)
    graph = build_graph(clients)  # type: ignore[arg-type]
    final = AgentState.model_validate(asyncio.run(graph.ainvoke(initial)))

    assert final.paper is None
    assert "extract_failed" in final.review_reasons
    assert final.review_item_id == 42
    assert final.note_path is None
    called = {(s, t) for s, t, _ in clients.calls}
    assert ("review_queue", "enqueue") in called
    assert ("obsidian_vault", "write_note") not in called


def test_missing_field_routes_to_escalate(pdf_path, monkeypatch) -> None:
    """Paper extracts but is missing required fields → escalate, no note written."""
    import packages.agent.nodes.extract as extract_node

    # Only title, no abstract, no authors.
    monkeypatch.setattr(
        extract_node,
        "claude_extract",
        lambda _p: Paper(title=ExtractedField(value="Only A Title", confidence=0.9)),
    )
    clients = FakeClients({("review_queue", "enqueue"): 99})
    initial = AgentState(arxiv_id="2506.11419", pdf_path=pdf_path)
    graph = build_graph(clients)  # type: ignore[arg-type]
    final = AgentState.model_validate(asyncio.run(graph.ainvoke(initial)))

    assert final.review_item_id == 99
    assert "missing:abstract" in final.review_reasons
    assert "missing:authors" in final.review_reasons
    assert final.note_path is None
