"""CLI: python -m packages.agent run <pdf> | resume <thread_id> <decision>

Subcommands:
  run     start a fresh agent run on a PDF. Thread id defaults to the PDF stem.
  resume  resume a paused thread with approve | reject | defer.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from packages.agent.clients import MCPClients
from packages.agent.graph import build_graph
from packages.agent.state import AgentState

load_dotenv()
app = typer.Typer(add_completion=False)

VALID_DECISIONS = ("approve", "reject", "defer")


def _dsn() -> str:
    user = os.environ.get("POSTGRES_USER", "abstracted")
    pwd = os.environ.get("POSTGRES_PASSWORD", "abstracted")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "abstracted")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def _print(final_raw: dict[str, Any] | Any, thread_id: str) -> None:
    """Render a final or interrupted state to the console."""
    # Interrupted runs include __interrupt__ from langgraph; surface that.
    if isinstance(final_raw, dict) and "__interrupt__" in final_raw:
        interrupts = final_raw["__interrupt__"]
        typer.echo(f"thread_id:   {thread_id}")
        typer.echo("status:      paused (awaiting review)")
        for it in interrupts:
            v = getattr(it, "value", it)
            typer.echo(f"interrupt:   {v}")
        typer.echo(
            f"resume with: uv run python -m packages.agent resume {thread_id} "
            f"<approve|reject|defer>"
        )
        return

    final = AgentState.model_validate(final_raw)
    typer.echo(f"thread_id:   {thread_id}")
    if final.review_item_id is not None and final.review_decision != "approve":
        typer.echo(f"escalated:   review item #{final.review_item_id}")
        typer.echo(f"reasons:     {', '.join(final.review_reasons)}")
        if final.review_decision:
            typer.echo(f"decision:    {final.review_decision}")
    else:
        typer.echo(f"note:        {final.note_path}")
        typer.echo(f"code urls:   {len(final.code_urls)}")
        typer.echo(f"related:     {len(final.related_papers)}")
    if final.errors:
        typer.echo("errors:", err=True)
        for e in final.errors:
            typer.echo(f"  - {e}", err=True)


async def _run(pdf_path: Path, thread_id: str) -> None:
    initial = AgentState(arxiv_id=pdf_path.stem, pdf_path=pdf_path)
    config = {"configurable": {"thread_id": thread_id}}
    async with (
        AsyncPostgresSaver.from_conn_string(_dsn()) as checkpointer,
        MCPClients() as clients,
    ):
        await checkpointer.setup()
        graph = build_graph(clients, checkpointer=checkpointer)
        final = await graph.ainvoke(initial, config=config)  # type: ignore[arg-type]
    _print(final, thread_id)


async def _resume(thread_id: str, decision: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    async with (
        AsyncPostgresSaver.from_conn_string(_dsn()) as checkpointer,
        MCPClients() as clients,
    ):
        graph = build_graph(clients, checkpointer=checkpointer)
        final = await graph.ainvoke(Command(resume=decision), config=config)  # type: ignore[arg-type]
    _print(final, thread_id)


@app.command()
def run(
    pdf_path: Path = typer.Argument(..., help="Path to PDF on disk."),
    thread_id: str | None = typer.Option(None, help="Thread id (default: PDF stem)."),
) -> None:
    if not pdf_path.exists():
        typer.echo(f"No such file: {pdf_path}", err=True)
        sys.exit(1)
    asyncio.run(_run(pdf_path, thread_id or pdf_path.stem))


@app.command()
def resume(
    thread_id: str = typer.Argument(..., help="Thread id from a paused run."),
    decision: str = typer.Argument(..., help="approve | reject | defer"),
) -> None:
    if decision not in VALID_DECISIONS:
        typer.echo(f"decision must be one of {VALID_DECISIONS}", err=True)
        sys.exit(1)
    asyncio.run(_resume(thread_id, decision))


if __name__ == "__main__":
    app()
