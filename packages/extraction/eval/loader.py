"""Dataset loader: pairs of <id>.pdf + <id>.json under data/eval/<dataset>/."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from packages.extraction.schema import Paper


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
