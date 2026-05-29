"""Photometric augmentation: render PDF pages to images with varied noise.

bboxes are unchanged (no geometric transform), so labeled JSON is copied as-is.
Per paper, produces N augmented variants. Output goes to data/labeled_aug/.
"""

from __future__ import annotations

import io
import random
import shutil
from pathlib import Path

import pymupdf  # type: ignore[import-untyped]
import typer
from PIL import Image, ImageEnhance, ImageFilter

LABELED_DIR = Path("data/labeled")
RAW_DIR = Path("data/raw")
AUG_DIR = Path("data/labeled_aug")
RENDER_DPI = 150

app = typer.Typer(add_completion=False)


def _render_pages(pdf_path: Path, dpi: int = RENDER_DPI) -> list[Image.Image]:
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
    return pages


def _photometric(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
    # JPEG round-trip introduces compression artifacts.
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(50, 90))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@app.command()
def main(
    n_aug: int = typer.Option(2, help="Augmented variants per paper."),
    seed: int = typer.Option(1337),
) -> None:
    rng = random.Random(seed)
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    targets = sorted(LABELED_DIR.glob("*.json"))
    if not targets:
        raise typer.BadParameter(f"No labeled JSONs in {LABELED_DIR}")
    n_done = 0
    for jpath in targets:
        aid = jpath.stem
        pdf = RAW_DIR / f"{aid}.pdf"
        if not pdf.exists():
            typer.echo(f"skip {aid}: missing PDF", err=True)
            continue
        try:
            pages = _render_pages(pdf)
        except Exception as e:
            typer.echo(f"skip {aid}: render failed: {e}", err=True)
            continue
        for k in range(n_aug):
            out_dir = AUG_DIR / f"{aid}_aug{k}"
            out_dir.mkdir(exist_ok=True)
            for i, page_img in enumerate(pages):
                augmented = _photometric(page_img, rng)
                augmented.save(out_dir / f"page_{i:03d}.png")
            shutil.copy2(jpath, AUG_DIR / f"{aid}_aug{k}.json")
        n_done += 1
        typer.echo(f"augmented {aid} ({len(pages)} pages × {n_aug})")
    typer.echo(f"done: {n_done} papers → {AUG_DIR}")


if __name__ == "__main__":
    app()
