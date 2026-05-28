"""CLI: python -m packages.extraction.eval --model <name> --dataset eval/v1"""

from pathlib import Path

import typer

from packages.extraction.eval.loader import load_dataset
from packages.extraction.eval.registry import get as get_model
from packages.extraction.eval.report import to_markdown
from packages.extraction.eval.scorer import aggregate, score_document

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option(..., help="Registered model name."),
    dataset: str = typer.Option(..., help="Dataset path relative to data/ (e.g. eval/v1)."),
    out: Path | None = typer.Option(None, help="Write markdown report here. Default: stdout."),
) -> None:
    extract = get_model(model)
    root = Path("data") / dataset
    scores = []
    for ex in load_dataset(root):
        pred = extract(ex.pdf_path)
        scores.append(score_document(ex.id, pred, ex.ground_truth))
    report = aggregate(scores)
    md = to_markdown(report, model_name=model, dataset_name=dataset)
    if out:
        out.write_text(md)
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(md)


if __name__ == "__main__":
    app()
