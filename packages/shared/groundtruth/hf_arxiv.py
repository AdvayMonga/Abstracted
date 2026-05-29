"""arXiv metadata via Hugging Face dataset (librarian-bots/arxiv-metadata-snapshot).

No API rate limits — streams the parquet snapshot directly. Used in place of
the arxiv.org API. PDFs still come from arxiv.org/pdf/<id>.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from datasets import load_dataset  # type: ignore[import-untyped]

from packages.shared.groundtruth.arxiv import CACHE_DIR, ArxivMeta

DATASET_ID = "librarian-bots/arxiv-metadata-snapshot"

_WS = re.compile(r"\s+")


def _split_authors(authors_field: str | None, parsed: list | None) -> list[str]:
    """Prefer authors_parsed (structured); fall back to splitting the string."""
    if parsed:
        out = []
        for entry in parsed:
            # Format is [last, first, suffix]; reassemble as "first last".
            if not entry:
                continue
            last = entry[0] if len(entry) > 0 else ""
            first = entry[1] if len(entry) > 1 else ""
            name = f"{first} {last}".strip()
            if name:
                out.append(name)
        if out:
            return out
    if not authors_field:
        return []
    # The raw string uses "Alice, Bob, and Charlie" patterns.
    parts = re.split(r",\s*|\s+and\s+", authors_field)
    return [_WS.sub(" ", p).strip() for p in parts if p.strip()]


def _row_to_meta(row: dict) -> ArxivMeta:
    return ArxivMeta(
        arxiv_id=row["id"],
        title=_WS.sub(" ", row["title"]).strip(),
        authors=_split_authors(row.get("authors"), row.get("authors_parsed")),
        abstract=_WS.sub(" ", row["abstract"]).strip(),
        categories=list((row.get("categories") or "").split()),
    )


def _cache_write(meta: ArxivMeta) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{meta.arxiv_id.replace('/', '_')}.json"
    path.write_text(json.dumps(meta.__dict__))


def _cache_read(arxiv_id: str) -> ArxivMeta | None:
    path = CACHE_DIR / f"{arxiv_id.replace('/', '_')}.json"
    if not path.exists():
        return None
    return ArxivMeta(**json.loads(path.read_text()))


def sample(
    categories: list[str],
    n: int,
    seed: int = 1337,
    year_min: int | None = 2018,
    pool_size: int = 5000,
) -> list[ArxivMeta]:
    """Stream the HF dataset, filter by category/year, reservoir-sample N papers.

    Stops streaming once we've collected `pool_size` candidates (else we'd read 3M rows).
    """
    wanted = set(categories)
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    pool: list[ArxivMeta] = []
    for row in ds:
        cats = (row.get("categories") or "").split()
        if not any(c in wanted for c in cats):
            continue
        if year_min is not None:
            ud = row.get("update_date")
            year = getattr(ud, "year", None)
            if year is None and isinstance(ud, str):
                m = re.match(r"(\d{4})", ud)
                year = int(m.group(1)) if m else None
            if year is None or year < year_min:
                continue
        pool.append(_row_to_meta(row))
        if len(pool) >= pool_size:
            break
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = pool[:n]
    for m in chosen:
        _cache_write(m)
    return chosen


def fetch_metadata(arxiv_id: str) -> ArxivMeta:
    """Load metadata from cache (must have been pre-populated via `sample`)."""
    cached = _cache_read(arxiv_id)
    if cached is None:
        raise LookupError(
            f"No HF cache for {arxiv_id}. Run `sample()` first or fall back to API."
        )
    return cached


def download_pdf(arxiv_id: str, dest_dir: Path) -> Path:
    """PDF download piggybacks on the existing API module."""
    from packages.shared.groundtruth.arxiv import download_pdf as api_download
    return api_download(arxiv_id, dest_dir)
