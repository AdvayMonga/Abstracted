"""CLI: python -m packages.extraction.eval --model <name> [--split val | --dataset eval/v1]"""

from pathlib import Path

import typer
from dotenv import load_dotenv

from packages.extraction.eval.loader import load_dataset, load_split
from packages.extraction.eval.registry import get as get_model
from packages.extraction.eval.report import to_markdown
from packages.extraction.eval.scorer import aggregate, score_document

load_dotenv()

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option(..., help="Registered model name."),
    split: str | None = typer.Option(None, help="Split name under data/splits/ (e.g. val)."),
    dataset: str | None = typer.Option(None, help="Legacy: path under data/ (e.g. eval/v1)."),
    limit: int | None = typer.Option(None, help="Cap number of examples (split mode)."),
    out: Path | None = typer.Option(None, help="Write markdown report here. Default: stdout."),
) -> None:
    if (split is None) == (dataset is None):
        raise typer.BadParameter("Pass exactly one of --split or --dataset.")
    extract = get_model(model)
    examples = (
        load_split(split, limit=limit) if split else load_dataset(Path("data") / dataset)  # type: ignore[arg-type]
    )
    scores = []
    for ex in examples:
        pred = extract(ex.pdf_path)
        scores.append(score_document(ex.id, pred, ex.ground_truth))
    report = aggregate(scores)
    label = f"split:{split}" if split else f"dataset:{dataset}"
    md = to_markdown(report, model_name=model, dataset_name=label)
    if out:
        out.write_text(md)
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(md)


if __name__ == "__main__":
    app()
