"""Graph-level test with a fake MCPClients. No subprocesses, no API calls."""

from __future__ import annotations

import asyncio
import os
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


def test_graph_happy_path(pdf_path, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACTED_VAULT_DIR", str(tmp_path / "vault"))

    # Bypass the real Claude call.
    import packages.agent.nodes.extract as extract_node

    def fake_extract(_path):
        return Paper(
            title=ExtractedField(value="Fake Paper", confidence=0.9),
            authors=[ExtractedField(value="A. Author", confidence=0.9)],
            abstract=ExtractedField(value="Test abstract.", confidence=0.9),
        )

    monkeypatch.setattr(extract_node, "claude_extract", fake_extract)

    note_path = str(tmp_path / "vault" / "Fake Paper (2506.11419).md")
    # The write_note tool will be called via FakeClients; have it return the
    # path it would have written so the node can record it in state.
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
    final_raw = asyncio.run(graph.ainvoke(initial))
    final = AgentState.model_validate(final_raw)

    assert final.paper is not None
    assert final.paper.title is not None
    assert final.paper.title.value == "Fake Paper"
    assert len(final.related_papers) == 2
    assert final.code_urls == ["https://github.com/foo/bar"]  # deduped
    assert final.note_path == note_path
    assert final.errors == []
    called = {(s, t) for s, t, _ in clients.calls}
    assert ("semantic_scholar", "recommendations") in called
    assert ("github_lookup", "scan_pdf") in called
    assert ("obsidian_vault", "write_note") in called


def test_extract_failure_records_error(pdf_path, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACTED_VAULT_DIR", str(tmp_path / "vault"))
    import packages.agent.nodes.extract as extract_node

    def boom(_path):
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(extract_node, "claude_extract", boom)
    clients = FakeClients({})
    initial = AgentState(arxiv_id="2506.11419", pdf_path=pdf_path)
    graph = build_graph(clients)  # type: ignore[arg-type]
    final = AgentState.model_validate(asyncio.run(graph.ainvoke(initial)))
    assert any("extract:" in e for e in final.errors)
    assert final.paper is None
    # write_note should fail cleanly because there's no paper.
    assert any("write_note" in e for e in final.errors)


# Tells pyright about a real test name to silence "imported but unused".
_ = os.environ
