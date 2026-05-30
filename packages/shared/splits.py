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
    """Partition ids into train/val/test/held_out, stratified by primary category.

    Within each category, distribute counts by ratio with floor, then assign
    leftover items to the buckets with the largest fractional remainders. This
    keeps the global ratio close to target instead of dumping all remainders
    into one bucket.
    """
    by_cat: dict[str, list[str]] = defaultdict(list)
    for aid in ids:
        by_cat[_primary_category(aid)].append(aid)

    out: dict[str, list[str]] = {k: [] for k in SPLIT_RATIOS}
    rng = random.Random(seed)
    bucket_names = list(SPLIT_RATIOS.keys())
    for cat_ids in by_cat.values():
        shuffled = cat_ids[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        # Compute floor counts and fractional remainders per bucket.
        raw = {b: n * SPLIT_RATIOS[b] for b in bucket_names}
        counts = {b: int(raw[b]) for b in bucket_names}
        leftover = n - sum(counts.values())
        # Distribute leftovers by largest fractional part first (Hamilton method).
        fracs = sorted(bucket_names, key=lambda b: raw[b] - counts[b], reverse=True)
        for b in fracs[:leftover]:
            counts[b] += 1
        # Slice.
        idx = 0
        for b in bucket_names:
            out[b].extend(shuffled[idx : idx + counts[b]])
            idx += counts[b]
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
