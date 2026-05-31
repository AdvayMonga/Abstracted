"""Smoke tests for the semantic_scholar MCP server."""

from __future__ import annotations

from packages.tools.semantic_scholar.server import lookup, mcp, recommendations


def test_tools_registered() -> None:
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert {"lookup", "recommendations"} <= names


def test_callable() -> None:
    assert callable(lookup)
    assert callable(recommendations)
