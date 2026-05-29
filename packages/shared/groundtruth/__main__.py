"""CLI: build ground-truth Paper JSONs for arXiv papers.

  python -m packages.shared.groundtruth fetch --category cs.LG --n 5
  python -m packages.shared.groundtruth build --id 2403.12345 --id 2403.67890
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from packages.shared.groundtruth import arxiv as arxiv_api
from packages.shared.groundtruth import hf_arxiv
from packages.shared.groundtruth.pipeline import build

app = typer.Typer(add_completion=False)

RAW_DIR = Path("data/raw")
LABELED_DIR = Path("data/labeled")
MANIFEST = RAW_DIR / "manifest.csv"


@app.command()
def fetch(
    category: list[str] = typer.Option(
        ["cs.LG", "cs.CL", "cs.AI", "cs.SY", "cs.DC"],
        help="arXiv categories to draw from.",
    ),
    n: int = typer.Option(5, help="Total papers to fetch."),
) -> None:
    """Search arXiv, download N PDFs, write manifest.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ids = arxiv_api.search(category, max_results=n)
    rows = ["arxiv_id,pdf_path"]
    for aid in ids:
        try:
            path = arxiv_api.download_pdf(aid, RAW_DIR)
            rows.append(f"{aid},{path}")
            typer.echo(f"fetched {aid}")
        except Exception as e:
            typer.echo(f"skip {aid}: {e}", err=True)
    MANIFEST.write_text("\n".join(rows) + "\n")
    typer.echo(f"wrote {MANIFEST}")


@app.command()
def fetch_hf(
    category: list[str] = typer.Option(
        ["cs.LG", "cs.CL", "cs.AI", "cs.SY", "cs.DC"],
        help="arXiv categories to filter on.",
    ),
    n: int = typer.Option(20, help="Total papers to sample."),
    year_min: int = typer.Option(2018, help="Min update year."),
    pool_size: int = typer.Option(2000, help="HF rows scanned before sampling."),
    seed: int = typer.Option(1337),
) -> None:
    """Sample papers from the HF metadata snapshot, then download their PDFs."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    typer.echo(f"sampling {n} papers from HF (pool={pool_size}, year>={year_min})...")
    metas = hf_arxiv.sample(category, n=n, seed=seed, year_min=year_min, pool_size=pool_size)
    typer.echo(f"got {len(metas)} candidates; downloading PDFs...")
    rows = ["arxiv_id,pdf_path"]
    for m in metas:
        try:
            path = hf_arxiv.download_pdf(m.arxiv_id, RAW_DIR)
            rows.append(f"{m.arxiv_id},{path}")
            typer.echo(f"fetched {m.arxiv_id}")
        except Exception as e:
            typer.echo(f"skip {m.arxiv_id}: {e}", err=True)
    MANIFEST.write_text("\n".join(rows) + "\n")
    typer.echo(f"wrote {MANIFEST}")


@app.command()
def build_cmd(
    id: list[str] = typer.Option(None, help="arxiv_ids. If omitted, uses manifest.csv."),
    grobid_url: str = typer.Option("http://localhost:8070/api/processFulltextDocument"),
) -> None:
    """Run the ground-truth pipeline; write data/labeled/<id>.json."""
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    if not id:
        if not MANIFEST.exists():
            raise typer.BadParameter("No --id given and no manifest.csv. Run `fetch` first.")
        rows = MANIFEST.read_text().strip().splitlines()[1:]
        targets = []
        for row in rows:
            aid, pdf = row.split(",", 1)
            targets.append((aid, Path(pdf)))
    else:
        targets = [(aid, RAW_DIR / f"{aid.replace('/', '_')}.pdf") for aid in id]

    summary = []
    for aid, pdf_path in targets:
        if not pdf_path.exists():
            typer.echo(f"skip {aid}: missing {pdf_path}", err=True)
            continue
        res = build(aid, pdf_path, grobid_url=grobid_url)
        out = LABELED_DIR / f"{aid.replace('/', '_')}.json"
        out.write_text(res.paper.model_dump_json(indent=2))
        p = res.paper
        summary.append(
            {
                "arxiv_id": aid,
                "title": bool(p.title and p.title.value),
                "title_bbox": bool(p.title and p.title.bbox),
                "authors": len(p.authors),
                "abstract": bool(p.abstract and p.abstract.value),
                "abstract_bbox": bool(p.abstract and p.abstract.bbox),
                "methods": bool(p.methods and p.methods.value),
                "citations": len(p.citations),
                "tools_code": len(p.tools_code),
                "key_results": len(p.key_results),
                "errors": res.errors,
            }
        )
        typer.echo(f"built {aid} → {out}")

    typer.echo("\n=== summary ===")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
