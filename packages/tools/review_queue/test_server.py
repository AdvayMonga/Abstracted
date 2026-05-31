"""Tests for review_queue. Live tests require a running Postgres on $POSTGRES_HOST."""

from __future__ import annotations

import os

import psycopg
import pytest
from dotenv import load_dotenv

from packages.tools.review_queue import server

load_dotenv()


def _pg_reachable() -> bool:
    try:
        with psycopg.connect(server._dsn(), connect_timeout=2):
            return True
    except Exception:
        return False


def test_tools_registered() -> None:
    names = {t.name for t in server.mcp._tool_manager.list_tools()}
    assert {"enqueue", "get", "list_items", "resolve"} <= names


@pytest.mark.skipif(
    not _pg_reachable(), reason="Postgres not reachable; start docker compose first."
)
def test_full_lifecycle() -> None:
    paper_id = f"test_{os.getpid()}"
    item_id = server.enqueue(paper_id, "low_conf:title", {"field": "title", "score": 0.3})
    assert isinstance(item_id, int) and item_id > 0

    got = server.get(item_id)
    assert got is not None
    assert got["paper_id"] == paper_id
    assert got["status"] == "pending"
    assert got["payload"]["field"] == "title"

    pending = [i for i in server.list_items(status="pending") if i["paper_id"] == paper_id]
    assert any(i["id"] == item_id for i in pending)

    resolved = server.resolve(item_id, "approved", reviewer_note="LGTM")
    assert resolved["status"] == "approved"
    assert resolved["resolved_at"] is not None
    assert resolved["reviewer_note"] == "LGTM"

    # cleanup
    with psycopg.connect(server._dsn(), autocommit=True) as c, c.cursor() as cur:
        cur.execute("DELETE FROM review_items WHERE paper_id = %s", (paper_id,))


def test_resolve_rejects_bad_status() -> None:
    with pytest.raises(ValueError):
        server.resolve(0, "not-a-status")
