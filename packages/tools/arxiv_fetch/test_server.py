"""Smoke tests for the arxiv_fetch MCP server.

We don't speak the MCP protocol here — we just confirm the tool functions are
registered with the FastMCP instance and that they're callable as plain Python.
"""

from __future__ import annotations

from packages.tools.arxiv_fetch.server import download_pdf, fetch_metadata, mcp, search


def test_tools_registered() -> None:
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert {"fetch_metadata", "download_pdf", "search"} <= names


def test_functions_are_plain_callables() -> None:
    # FastMCP keeps the wrapped fn callable directly.
    assert callable(fetch_metadata)
    assert callable(download_pdf)
    assert callable(search)
