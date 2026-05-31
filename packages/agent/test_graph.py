"""Graph-level tests with a fake MCPClients + in-memory checkpointer."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pymupdf  # type: ignore[import-untyped]
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from packages.agent.graph import build_graph
from packages.agent.state import AgentState
from packages.extraction.schema import ExtractedField, Paper


class FakeClients:
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


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def test_graph_happy_path(pdf_path, tmp_path, monkeypatch) -> None:
    """No escalation, no interrupt. Pipeline writes a note."""
    monkeypatch.setenv("ABSTRACTED_VAULT_DIR", str(tmp_path / "vault"))
    import packages.agent.nodes.extract as extract_node

    monkeypatch.setattr(extract_node, "claude_extract", lambda _p: _good_paper())

    note_path = str(tmp_path / "vault" / "Fake Paper (2506.11419).md")
    clients = FakeClients(
        {
            ("semantic_scholar", "recommendations"): [{"title": "R1", "year": 2024}],
            ("github_lookup", "scan_pdf"): ["https://github.com/foo/bar"],
            ("github_lookup", "pwc_lookup"): None,
            ("obsidian_vault", "write_note"): note_path,
        }
    )
    saver = InMemorySaver()
    graph = build_graph(clients, checkpointer=saver)  # type: ignore[arg-type]
    initial = AgentState(arxiv_id="2506.11419", pdf_path=pdf_path)
    final = AgentState.model_validate(asyncio.run(graph.ainvoke(initial, _config("t1"))))

    assert final.note_path == note_path
    assert final.review_reasons == []
    assert final.review_item_id is None
    assert final.errors == []
    called = {(s, t) for s, t, _ in clients.calls}
    assert ("obsidian_vault", "write_note") in called
    assert ("review_queue", "enqueue") not in called


def test_escalation_pauses_then_rejects(pdf_path, monkeypatch) -> None:
    """Extract fails → escalate enqueues, interrupts; resume(reject) ends without note."""
    import packages.agent.nodes.extract as extract_node

    monkeypatch.setattr(
        extract_node, "claude_extract", lambda _p: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    clients = FakeClients(
        {
            ("review_queue", "enqueue"): 42,
            ("review_queue", "resolve"): {"id": 42, "status": "rejected"},
        }
    )
    saver = InMemorySaver()
    graph = build_graph(clients, checkpointer=saver)  # type: ignore[arg-type]
    cfg = _config("paused-1")

    # First invocation: hits interrupt(), returns with __interrupt__ present.
    first = asyncio.run(
        graph.ainvoke(AgentState(arxiv_id="2506.11419", pdf_path=pdf_path), cfg)
    )
    assert "__interrupt__" in first
    interrupt_obj = first["__interrupt__"][0]
    assert interrupt_obj.value["kind"] == "review_needed"
    assert interrupt_obj.value["review_item_id"] == 42
    assert "extract_failed" in interrupt_obj.value["reasons"]

    # Resume with reject.
    final_raw = asyncio.run(graph.ainvoke(Command(resume="reject"), cfg))
    final = AgentState.model_validate(final_raw)
    assert final.review_decision == "reject"
    assert final.review_item_id == 42
    assert final.note_path is None
    called = {(s, t) for s, t, _ in clients.calls}
    assert ("review_queue", "resolve") in called
    assert ("obsidian_vault", "write_note") not in called


def test_escalation_then_approve_continues_pipeline(pdf_path, tmp_path, monkeypatch) -> None:
    """Missing-field escalation → resume(approve) → enrich + write_note both run."""
    monkeypatch.setenv("ABSTRACTED_VAULT_DIR", str(tmp_path / "vault"))
    import packages.agent.nodes.extract as extract_node

    # Title only — triggers missing:abstract, missing:authors.
    monkeypatch.setattr(
        extract_node,
        "claude_extract",
        lambda _p: Paper(title=ExtractedField(value="Title Only", confidence=0.9)),
    )
    note_path = str(tmp_path / "vault" / "Title Only (2506.11419).md")
    clients = FakeClients(
        {
            ("review_queue", "enqueue"): 7,
            ("review_queue", "resolve"): {"id": 7, "status": "approved"},
            ("semantic_scholar", "recommendations"): [],
            ("github_lookup", "scan_pdf"): [],
            ("github_lookup", "pwc_lookup"): None,
            ("obsidian_vault", "write_note"): note_path,
        }
    )
    saver = InMemorySaver()
    graph = build_graph(clients, checkpointer=saver)  # type: ignore[arg-type]
    cfg = _config("paused-2")

    first = asyncio.run(
        graph.ainvoke(AgentState(arxiv_id="2506.11419", pdf_path=pdf_path), cfg)
    )
    assert "__interrupt__" in first

    final = AgentState.model_validate(
        asyncio.run(graph.ainvoke(Command(resume="approve"), cfg))
    )
    assert final.review_decision == "approve"
    assert final.note_path == note_path
    called = {(s, t) for s, t, _ in clients.calls}
    assert ("review_queue", "resolve") in called
    assert ("obsidian_vault", "write_note") in called
