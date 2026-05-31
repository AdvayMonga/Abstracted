"""CLI: python -m packages.agent <pdf_path>

The arxiv id is taken from the PDF filename (e.g. data/raw/2506.11419.pdf).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from packages.agent.clients import MCPClients
from packages.agent.graph import build_graph
from packages.agent.state import AgentState

load_dotenv()

app = typer.Typer(add_completion=False)


async def _run(pdf_path: Path) -> AgentState:
    arxiv_id = pdf_path.stem
    initial = AgentState(arxiv_id=arxiv_id, pdf_path=pdf_path)
    async with MCPClients() as clients:
        graph = build_graph(clients)
        final = await graph.ainvoke(initial)
    return AgentState.model_validate(final)


@app.command()
def main(pdf_path: Path = typer.Argument(..., help="Path to PDF on disk.")) -> None:
    if not pdf_path.exists():
        typer.echo(f"No such file: {pdf_path}", err=True)
        sys.exit(1)
    final = asyncio.run(_run(pdf_path))
    typer.echo(f"arxiv_id:    {final.arxiv_id}")
    if final.review_item_id is not None:
        typer.echo(f"escalated:   review item #{final.review_item_id}")
        typer.echo(f"reasons:     {', '.join(final.review_reasons)}")
    else:
        typer.echo(f"note:        {final.note_path}")
        typer.echo(f"code urls:   {len(final.code_urls)}")
        typer.echo(f"related:     {len(final.related_papers)}")
    if final.errors:
        typer.echo("errors:", err=True)
        for e in final.errors:
            typer.echo(f"  - {e}", err=True)


if __name__ == "__main__":
    app()
