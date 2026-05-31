"""review_queue MCP server.

A simple persistent escalation queue backed by Postgres. The agent enqueues
items it can't confidently handle (low extraction confidence, ambiguous
disambiguation, missing field); a human resolves them later.

Schema (auto-created on first connect):

  review_items (
    id           bigserial primary key,
    paper_id     text not null,
    reason       text not null,
    payload      jsonb not null,
    status       text not null default 'pending',
    reviewer_note text,
    created_at   timestamptz not null default now(),
    resolved_at  timestamptz
  )

Tools:
  enqueue(paper_id, reason, payload) -> id
  get(item_id) -> item
  list(status, limit) -> list[item]
  resolve(item_id, status, reviewer_note) -> item
"""

from __future__ import annotations

import json
import os
from typing import Any

import psycopg
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("review_queue")

VALID_STATUSES = {"pending", "approved", "rejected", "deferred"}
RESOLVED_STATUSES = {"approved", "rejected"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id            bigserial PRIMARY KEY,
    paper_id      text NOT NULL,
    reason        text NOT NULL,
    payload       jsonb NOT NULL,
    status        text NOT NULL DEFAULT 'pending',
    reviewer_note text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    resolved_at   timestamptz
);
CREATE INDEX IF NOT EXISTS review_items_status_idx ON review_items (status);
CREATE INDEX IF NOT EXISTS review_items_paper_idx  ON review_items (paper_id);
"""

_schema_ready = False


def _dsn() -> str:
    user = os.environ.get("POSTGRES_USER", "abstracted")
    pwd = os.environ.get("POSTGRES_PASSWORD", "abstracted")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "abstracted")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def _connect() -> psycopg.Connection:
    global _schema_ready
    conn = psycopg.connect(_dsn(), autocommit=True)
    if not _schema_ready:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)
        _schema_ready = True
    return conn


def _row_to_dict(row: tuple) -> dict[str, Any]:
    keys: tuple[str, ...] = (
        "id",
        "paper_id",
        "reason",
        "payload",
        "status",
        "reviewer_note",
        "created_at",
        "resolved_at",
    )
    out: dict[str, Any] = dict(zip(keys, row, strict=True))
    out["created_at"] = out["created_at"].isoformat() if out["created_at"] else None
    out["resolved_at"] = out["resolved_at"].isoformat() if out["resolved_at"] else None
    return out


@mcp.tool()
def enqueue(paper_id: str, reason: str, payload: dict) -> int:
    """Add a low-confidence item to the review queue. Returns the new id.

    Args:
        paper_id: arXiv id (or any opaque paper identifier).
        reason: short string explaining why this needs review.
        payload: arbitrary JSON-serializable dict (the thing under review).
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_items (paper_id, reason, payload) "
            "VALUES (%s, %s, %s::jsonb) RETURNING id",
            (paper_id, reason, json.dumps(payload)),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


@mcp.tool()
def get(item_id: int) -> dict[str, Any] | None:
    """Fetch a single review item by id. Returns None if not found."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, paper_id, reason, payload, status, reviewer_note, "
            "       created_at, resolved_at "
            "FROM review_items WHERE id = %s",
            (item_id,),
        )
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


@mcp.tool()
def list_items(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """List review items, optionally filtered by status. Newest first."""
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status!r}. Allowed: {sorted(VALID_STATUSES)}")
    where = "WHERE status = %s" if status else ""
    params: tuple = (status, limit) if status else (limit,)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, paper_id, reason, payload, status, reviewer_note, "
            "       created_at, resolved_at "
            f"FROM review_items {where} ORDER BY created_at DESC LIMIT %s",
            params,
        )
        return [_row_to_dict(r) for r in cur.fetchall()]


@mcp.tool()
def resolve(item_id: int, status: str, reviewer_note: str | None = None) -> dict[str, Any]:
    """Set an item's status. Stamps resolved_at when moving to a terminal state."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status!r}. Allowed: {sorted(VALID_STATUSES)}")
    resolved_clause = "resolved_at = now()" if status in RESOLVED_STATUSES else "resolved_at = NULL"
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE review_items SET status = %s, reviewer_note = %s, {resolved_clause} "
            "WHERE id = %s "
            "RETURNING id, paper_id, reason, payload, status, reviewer_note, "
            "          created_at, resolved_at",
            (status, reviewer_note, item_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"No review item with id={item_id}")
        return _row_to_dict(row)
