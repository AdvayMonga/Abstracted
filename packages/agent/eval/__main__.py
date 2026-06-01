"""Agent eval CLI.

Run the agent on N papers from a split and write a pass/fail grader report.

  uv run python -m packages.agent.eval --split val --limit 5 --out AGENT_EVAL.md
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from packages.agent.clients import MCPClients
from packages.agent.eval.graders import GRADERS, grade
from packages.agent.graph import build_graph
from packages.agent.state import AgentState

load_dotenv()
app = typer.Typer(add_completion=False)

SPLITS_DIR = Path("data/splits")
RAW_DIR = Path("data/raw")


def _dsn() -> str:
    user = os.environ.get("POSTGRES_USER", "abstracted")
    pwd = os.environ.get("POSTGRES_PASSWORD", "abstracted")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "abstracted")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def _load_ids(split: str, limit: int | None) -> list[str]:
    f = SPLITS_DIR / f"{split}.txt"
    if not f.exists():
        raise FileNotFoundError(f"No split file: {f}")
    ids = [s.strip() for s in f.read_text().splitlines() if s.strip()]
    return ids[:limit] if limit else ids


async def _run_one(graph: Any, arxiv_id: str) -> tuple[AgentState, float]:
    pdf = RAW_DIR / f"{arxiv_id}.pdf"
    initial = AgentState(arxiv_id=arxiv_id, pdf_path=pdf)
    thread_id = f"eval-{arxiv_id}-{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.time()
    final_raw = await graph.ainvoke(initial, config)
    if "__interrupt__" in final_raw:
        # Treat an interrupt as a stop point; the run remains paused in pg.
        final = AgentState.model_validate(
            {
                **{k: v for k, v in final_raw.items() if k != "__interrupt__"},
                "review_reasons": final_raw.get("review_reasons", []),
            }
        )
    else:
        final = AgentState.model_validate(final_raw)
    return final, time.time() - t0


def _write_report(rows: list[dict[str, Any]], out: Path) -> None:
    grader_names = list(GRADERS.keys())
    totals = {g: 0 for g in grader_names}
    for r in rows:
        for g in grader_names:
            if r["scores"][g]:
                totals[g] += 1
    n = len(rows) or 1
    lines: list[str] = []
    lines.append("# Agent eval\n")
    lines.append(f"- documents: {n}")
    pct = {g: totals[g] / n for g in grader_names}
    overall = sum(pct.values()) / len(pct) if pct else 0.0
    lines.append(f"- overall pass-rate: {overall:.2%}\n")
    lines.append("## Pass rates by grader\n")
    lines.append("| grader | pass-rate |")
    lines.append("|---|---|")
    for g in grader_names:
        lines.append(f"| {g} | {pct[g]:.0%} ({totals[g]}/{n}) |")
    lines.append("")
    lines.append("## Per-document\n")
    header = "| arxiv_id | sec |" + " | ".join(grader_names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(grader_names)))
    for r in rows:
        cells = ["✓" if r["scores"][g] else "✗" for g in grader_names]
        lines.append(f"| {r['arxiv_id']} | {r['elapsed']:.1f}s | " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines) + "\n")


async def _main(split: str, limit: int | None, out: Path) -> None:
    ids = _load_ids(split, limit)
    vault = Path(os.environ.get("ABSTRACTED_VAULT_DIR", "data/vault"))
    typer.echo(f"running {len(ids)} papers from split={split}")
    rows: list[dict[str, Any]] = []
    async with (
        AsyncPostgresSaver.from_conn_string(_dsn()) as checkpointer,
        MCPClients() as clients,
    ):
        await checkpointer.setup()
        graph = build_graph(clients, checkpointer=checkpointer)
        for i, aid in enumerate(ids, 1):
            try:
                final, dt = await _run_one(graph, aid)
                scores = grade(final, vault)
            except Exception as e:
                final = AgentState(arxiv_id=aid, pdf_path=RAW_DIR / f"{aid}.pdf")
                final.errors.append(f"runner: {type(e).__name__}: {e}")
                scores = grade(final, vault)
                dt = 0.0
            rows.append({"arxiv_id": aid, "elapsed": dt, "scores": scores})
            passed = sum(scores.values())
            typer.echo(
                f"[{i}/{len(ids)}] {aid} {dt:.1f}s  {passed}/{len(scores)} passed",
                err=True,
            )
            sys.stderr.flush()
    _write_report(rows, out)
    typer.echo(f"wrote {out}")


@app.command()
def main(
    split: str = typer.Option("val", help="Split file under data/splits/."),
    limit: int | None = typer.Option(5, help="Cap the number of papers."),
    out: Path = typer.Option(Path("AGENT_EVAL.md"), help="Report path."),
) -> None:
    asyncio.run(_main(split, limit, out))


if __name__ == "__main__":
    app()
