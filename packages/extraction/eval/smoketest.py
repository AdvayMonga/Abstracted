"""Generate a tiny synthetic PDF + ground-truth JSON for smoke-testing the eval harness.

Run once: python -m packages.extraction.eval.smoketest
Then: python -m packages.extraction.eval --model pymupdf_regex --dataset eval/smoke
"""

from pathlib import Path

import pymupdf  # type: ignore[import-untyped]

from packages.extraction.schema import ExtractedField, Paper

OUT = Path("data/eval/smoke")
TITLE = "On the Foundations of Document Understanding"
AUTHORS = ["Ada Lovelace", "Alan Turing"]
ABSTRACT = (
    "We study the problem of extracting structured information from research papers. "
    "Our approach combines a vision-language model with a downstream agent. "
    "Experiments on a small held-out set indicate the approach is promising."
)


def build_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), TITLE, fontsize=20)
    page.insert_text((72, 130), ", ".join(AUTHORS), fontsize=11)
    page.insert_text((72, 180), "Abstract", fontsize=12)
    # Wrap the abstract body across a few lines.
    y = 200
    for line in [ABSTRACT[i : i + 90] for i in range(0, len(ABSTRACT), 90)]:
        page.insert_text((72, y), line, fontsize=10)
        y += 14
    page.insert_text((72, y + 20), "1. Introduction", fontsize=12)
    doc.save(path)
    doc.close()


def build_gt(path: Path) -> None:
    paper = Paper(
        title=ExtractedField(value=TITLE, confidence=1.0),
        authors=[ExtractedField(value=a, confidence=1.0) for a in AUTHORS],
        abstract=ExtractedField(value=ABSTRACT, confidence=1.0),
    )
    path.write_text(paper.model_dump_json(indent=2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_pdf(OUT / "smoke_001.pdf")
    build_gt(OUT / "smoke_001.json")
    print(f"Wrote smoke dataset to {OUT}")


if __name__ == "__main__":
    main()
