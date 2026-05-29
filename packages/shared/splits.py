"""Stratified 60/15/15/10 splits over labeled papers, keyed by arXiv category."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import typer

SPLIT_RATIOS = {"train": 0.60, "val": 0.15, "test": 0.15, "held_out": 0.10}
LABELED_DIR = Path("data/labeled")
CACHE_DIR = Path("data/cache/arxiv")
SPLITS_DIR = Path("data/splits")

app = typer.Typer(add_completion=False)


def _primary_category(arxiv_id: str) -> str:
    cache = CACHE_DIR / f"{arxiv_id.replace('/', '_')}.json"
    if not cache.exists():
        return "unknown"
    data = json.loads(cache.read_text())
    cats = data.get("categories") or []
    return cats[0] if cats else "unknown"


def stratified_split(
    ids: list[str], seed: int = 1337
) -> dict[str, list[str]]:
    """Partition ids into train/val/test/held_out, stratified by primary category."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    for aid in ids:
        by_cat[_primary_category(aid)].append(aid)

    out: dict[str, list[str]] = {k: [] for k in SPLIT_RATIOS}
    rng = random.Random(seed)
    for cat_ids in by_cat.values():
        shuffled = cat_ids[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        n_test = int(n * SPLIT_RATIOS["test"])
        out["train"].extend(shuffled[:n_train])
        out["val"].extend(shuffled[n_train : n_train + n_val])
        out["test"].extend(shuffled[n_train + n_val : n_train + n_val + n_test])
        out["held_out"].extend(shuffled[n_train + n_val + n_test :])
    return out


@app.command()
def main(seed: int = 1337) -> None:
    ids = sorted(p.stem for p in LABELED_DIR.glob("*.json"))
    if not ids:
        raise typer.BadParameter(f"No labeled JSONs in {LABELED_DIR}")
    splits = stratified_split(ids, seed=seed)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        (SPLITS_DIR / f"{name}.txt").write_text("\n".join(sorted(members)) + "\n")
        typer.echo(f"{name}: {len(members)}")


if __name__ == "__main__":
    app()
