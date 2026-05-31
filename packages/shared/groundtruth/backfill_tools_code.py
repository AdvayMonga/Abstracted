"""One-shot: rescan all PDFs in data/raw/ for github URLs, merge into labels."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from packages.extraction.schema import ExtractedField, Paper
from packages.shared.groundtruth.github_links import scan

CONF_PDF_SCAN = 0.9
RAW_DIR = Path("data/raw")
LABELED_DIR = Path("data/labeled")

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    label_files = sorted(LABELED_DIR.glob("*.json"))
    before = after = 0
    added = 0
    for lf in label_files:
        paper = Paper.model_validate_json(lf.read_text())
        had = len(paper.tools_code) > 0
        before += int(had)

        pdf = RAW_DIR / f"{lf.stem}.pdf"
        if not pdf.exists():
            after += int(had)
            continue

        urls = scan(pdf)
        existing = {f.value for f in paper.tools_code}
        new = [u for u in urls if u not in existing]
        if new:
            paper.tools_code.extend(
                ExtractedField(value=u, bbox=None, confidence=CONF_PDF_SCAN) for u in new
            )
            added += len(new)
            lf.write_text(json.dumps(paper.model_dump(mode="json"), indent=2) + "\n")

        after += int(len(paper.tools_code) > 0)

    typer.echo(f"papers with tools_code: {before} → {after} ({len(label_files)} total)")
    typer.echo(f"added {added} new URLs")


if __name__ == "__main__":
    app()
