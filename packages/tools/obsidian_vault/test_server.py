"""Tests for the obsidian_vault MCP server: real filesystem, temp vault."""

from __future__ import annotations

import pytest

from packages.tools.obsidian_vault import server


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("ABSTRACTED_VAULT_DIR", str(tmp_path))
    return tmp_path


def test_tools_registered() -> None:
    names = {t.name for t in server.mcp._tool_manager.list_tools()}
    assert {"write_note", "read_note", "append_to_note", "list_notes"} <= names


def test_write_and_read_with_frontmatter(vault) -> None:
    server.write_note(
        slug="FocalAD",
        body="# FocalAD\n\nNotes.",
        frontmatter={"arxiv_id": "2506.11419", "tags": ["autonomous-driving"]},
    )
    got = server.read_note("FocalAD")
    assert got["frontmatter"]["arxiv_id"] == "2506.11419"
    assert got["frontmatter"]["tags"] == ["autonomous-driving"]
    assert got["body"].startswith("# FocalAD")


def test_write_without_frontmatter(vault) -> None:
    server.write_note(slug="plain", body="just body")
    got = server.read_note("plain")
    assert got["frontmatter"] is None
    assert got["body"] == "just body"


def test_append(vault) -> None:
    server.write_note(slug="log", body="line 1")
    server.append_to_note("log", "line 2")
    got = server.read_note("log")
    assert "line 1" in got["body"] and "line 2" in got["body"]


def test_list_notes(vault) -> None:
    server.write_note(slug="a", body="x")
    server.write_note(slug="b", body="y")
    assert server.list_notes() == ["a", "b"]


def test_rejects_bad_slug(vault) -> None:
    with pytest.raises(ValueError):
        server.write_note(slug="../etc/passwd", body="x")


def test_read_missing_raises(vault) -> None:
    with pytest.raises(FileNotFoundError):
        server.read_note("does-not-exist")
