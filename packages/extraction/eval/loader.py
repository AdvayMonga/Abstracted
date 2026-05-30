"""Dataset loaders.

Two layouts are supported:
- `load_dataset(root)`: pairs of `<id>.pdf` + `<id>.json` under a single directory.
- `load_split(split)`: pulls ids from `data/splits/<split>.txt`, PDFs from `data/raw/`,
  ground truth from `data/labeled/`.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from packages.extraction.schema import Paper

RAW_DIR = Path("data/raw")
LABELED_DIR = Path("data/labeled")
SPLITS_DIR = Path("data/splits")


@dataclass(frozen=True)
class EvalExample:
    id: str
    pdf_path: Path
    ground_truth: Paper


def load_dataset(root: Path) -> Iterator[EvalExample]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    for pdf in sorted(root.glob("*.pdf")):
        gt_path = pdf.with_suffix(".json")
        if not gt_path.exists():
            continue
        gt = Paper.model_validate_json(gt_path.read_text())
        yield EvalExample(id=pdf.stem, pdf_path=pdf, ground_truth=gt)


def load_split(split: str, limit: int | None = None) -> Iterator[EvalExample]:
    """Yield examples listed in `data/splits/<split>.txt`."""
    split_file = SPLITS_DIR / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    if limit is not None:
        ids = ids[:limit]
    for aid in ids:
        pdf = RAW_DIR / f"{aid}.pdf"
        gt_path = LABELED_DIR / f"{aid}.json"
        if not pdf.exists() or not gt_path.exists():
            continue
        gt = Paper.model_validate_json(gt_path.read_text())
        yield EvalExample(id=aid, pdf_path=pdf, ground_truth=gt)
