"""Compute dataset statistics → data/STATS.md."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import pymupdf  # type: ignore[import-untyped]
import typer

from packages.extraction.schema import Paper

LABELED_DIR = Path("data/labeled")
RAW_DIR = Path("data/raw")
STATS_PATH = Path("data/STATS.md")
CACHE_DIR = Path("data/cache/arxiv")

app = typer.Typer(add_completion=False)


def _tok_count(s: str | None) -> int:
    return 0 if not s else len(s.split())


def _quartiles(xs: list[int]) -> tuple[int, int, int]:
    if not xs:
        return (0, 0, 0)
    s = sorted(xs)
    q = statistics.quantiles(s, n=4) if len(s) >= 4 else [s[0], s[len(s) // 2], s[-1]]
    return (int(q[0]), int(q[1]), int(q[2]))


def _primary_cat(arxiv_id: str) -> str:
    cache = CACHE_DIR / f"{arxiv_id.replace('/', '_')}.json"
    if not cache.exists():
        return "unknown"
    data = json.loads(cache.read_text())
    cats = data.get("categories") or []
    return cats[0] if cats else "unknown"


@app.command()
def main() -> None:
    papers = []
    for jpath in sorted(LABELED_DIR.glob("*.json")):
        try:
            papers.append((jpath.stem, Paper.model_validate_json(jpath.read_text())))
        except Exception as e:
            typer.echo(f"skip {jpath.name}: {e}", err=True)
    if not papers:
        raise typer.BadParameter(f"No labeled papers in {LABELED_DIR}")

    n = len(papers)
    title_tokens = [_tok_count(p.title.value) for _, p in papers if p.title]
    abstract_tokens = [_tok_count(p.abstract.value) for _, p in papers if p.abstract]
    methods_tokens = [_tok_count(p.methods.value) for _, p in papers if p.methods]
    n_authors = [len(p.authors) for _, p in papers]
    n_citations = [len(p.citations) for _, p in papers]
    n_keyresults = [len(p.key_results) for _, p in papers]

    presence = {
        "title": sum(1 for _, p in papers if p.title) / n,
        "title_bbox": sum(1 for _, p in papers if p.title and p.title.bbox) / n,
        "abstract": sum(1 for _, p in papers if p.abstract) / n,
        "abstract_bbox": sum(1 for _, p in papers if p.abstract and p.abstract.bbox) / n,
        "methods": sum(1 for _, p in papers if p.methods) / n,
        "tools_code": sum(1 for _, p in papers if p.tools_code) / n,
        "key_results": sum(1 for _, p in papers if p.key_results) / n,
    }

    cats = Counter(_primary_cat(aid) for aid, _ in papers)
    page_counts = []
    for aid, _ in papers:
        pdf = RAW_DIR / f"{aid}.pdf"
        if pdf.exists():
            with pymupdf.open(pdf) as doc:
                page_counts.append(doc.page_count)

    lines = ["# Dataset statistics", ""]
    lines.append(f"- documents: {n}")
    lines.append("")
    lines.append("## Category distribution")
    lines.append("")
    lines.append("| category | count |")
    lines.append("|---|---|")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {cnt} |")
    lines.append("")
    lines.append("## Field presence rates")
    lines.append("")
    lines.append("| field | present in |")
    lines.append("|---|---|")
    for f, rate in presence.items():
        lines.append(f"| {f} | {rate:.1%} |")
    lines.append("")
    lines.append("## Token length (Q1 / median / Q3)")
    lines.append("")
    lines.append("| field | Q1 | median | Q3 |")
    lines.append("|---|---|---|---|")
    for name, vals in [
        ("title", title_tokens),
        ("abstract", abstract_tokens),
        ("methods", methods_tokens),
    ]:
        q1, q2, q3 = _quartiles(vals)
        lines.append(f"| {name} | {q1} | {q2} | {q3} |")
    lines.append("")
    lines.append("## List-field sizes (Q1 / median / Q3)")
    lines.append("")
    lines.append("| field | Q1 | median | Q3 |")
    lines.append("|---|---|---|---|")
    for name, vals in [
        ("authors", n_authors),
        ("citations", n_citations),
        ("key_results", n_keyresults),
    ]:
        q1, q2, q3 = _quartiles(vals)
        lines.append(f"| {name} | {q1} | {q2} | {q3} |")
    if page_counts:
        q1, q2, q3 = _quartiles(page_counts)
        lines.append("")
        lines.append(f"## Page count: Q1={q1} median={q2} Q3={q3} max={max(page_counts)}")
    lines.append("")
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text("\n".join(lines))
    typer.echo(f"wrote {STATS_PATH}")


if __name__ == "__main__":
    app()
