"""Smoke tests for the github_lookup MCP server."""

from __future__ import annotations

from packages.tools.github_lookup.server import mcp, pwc_lookup, scan_pdf


def test_tools_registered() -> None:
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert {"scan_pdf", "pwc_lookup"} <= names


def test_callable() -> None:
    assert callable(scan_pdf)
    assert callable(pwc_lookup)
