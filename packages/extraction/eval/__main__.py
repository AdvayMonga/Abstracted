"""CLI: python -m packages.extraction.eval --model <name> [--split val | --dataset eval/v1]

Caches each prediction to data/cache/predictions/<model>/<id>.json so partial
progress survives a kill. Re-runs skip cached docs unless --no-cache is set.
Prints one line per document so you can see liveness.
"""

import sys
import time
from pathlib import Path

import typer
from dotenv import load_dotenv

from packages.extraction.eval.loader import load_dataset, load_split
from packages.extraction.eval.registry import get as get_model
from packages.extraction.eval.report import to_markdown
from packages.extraction.eval.scorer import aggregate, score_document
from packages.extraction.schema import Paper

load_dotenv()

app = typer.Typer(add_completion=False)
PREDICTION_CACHE = Path("data/cache/predictions")


@app.command()
def main(
    model: str = typer.Option(..., help="Registered model name."),
    split: str | None = typer.Option(None, help="Split name under data/splits/ (e.g. val)."),
    dataset: str | None = typer.Option(None, help="Legacy: path under data/ (e.g. eval/v1)."),
    limit: int | None = typer.Option(None, help="Cap number of examples (split mode)."),
    out: Path | None = typer.Option(None, help="Write markdown report here. Default: stdout."),
    no_cache: bool = typer.Option(False, help="Ignore cached predictions and re-run."),
) -> None:
    if (split is None) == (dataset is None):
        raise typer.BadParameter("Pass exactly one of --split or --dataset.")
    extract = get_model(model)
    examples = list(
        load_split(split, limit=limit) if split else load_dataset(Path("data") / dataset)  # type: ignore[arg-type]
    )
    cache_dir = PREDICTION_CACHE / model
    cache_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    total = len(examples)
    for i, ex in enumerate(examples, 1):
        cache_file = cache_dir / f"{ex.id}.json"
        t0 = time.time()
        if cache_file.exists() and not no_cache:
            pred = Paper.model_validate_json(cache_file.read_text())
            tag = "cached"
        else:
            try:
                pred = extract(ex.pdf_path)
                cache_file.write_text(pred.model_dump_json(indent=2))
                tag = "ok"
            except Exception as e:
                typer.echo(f"[{i}/{total}] {ex.id} FAIL ({type(e).__name__}: {e})", err=True)
                continue
        s = score_document(ex.id, pred, ex.ground_truth)
        scores.append(s)
        dt = time.time() - t0
        typer.echo(
            f"[{i}/{total}] {ex.id} {tag} {dt:.1f}s  "
            f"title={s.scalar['title']:.2f} cit_f1={s.lists['citations'].f1:.2f}",
            err=True,
        )
        sys.stderr.flush()

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
