"""Compose ground-truth Paper from arxiv + GROBID + PwC + key_results regex."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf  # type: ignore[import-untyped]

from packages.extraction.schema import ExtractedField, Paper
from packages.shared.groundtruth import arxiv, grobid, key_results, pwc
from packages.shared.groundtruth.locate import locate

CONF_ARXIV = 0.95
CONF_GROBID = 0.85
CONF_PWC = 0.9
CONF_REGEX = 0.5


@dataclass
class BuildResult:
    paper: Paper
    arxiv_id: str
    errors: list[str]


def build(arxiv_id: str, pdf_path: Path, grobid_url: str = grobid.GROBID_URL) -> BuildResult:
    errors: list[str] = []
    title = abstract = None
    authors: list[ExtractedField[str]] = []

    # 1. arXiv API
    try:
        meta = arxiv.fetch_metadata(arxiv_id)
        with pymupdf.open(pdf_path) as doc:
            title_bbox = locate(doc, meta.title)
            abstract_bbox = locate(doc, meta.abstract)
            author_bboxes = [locate(doc, a) for a in meta.authors]
        title = ExtractedField(value=meta.title, bbox=title_bbox, confidence=CONF_ARXIV)
        abstract = ExtractedField(value=meta.abstract, bbox=abstract_bbox, confidence=CONF_ARXIV)
        authors = [
            ExtractedField(value=a, bbox=bb, confidence=CONF_ARXIV)
            for a, bb in zip(meta.authors, author_bboxes, strict=True)
        ]
    except Exception as e:
        errors.append(f"arxiv: {e}")

    # 2. GROBID
    methods = None
    citations: list[ExtractedField] = []
    results_text = None
    try:
        gx = grobid.extract(pdf_path, base_url=grobid_url)
        if gx.methods_text:
            methods = ExtractedField(
                value=gx.methods_text, bbox=gx.methods_bbox, confidence=CONF_GROBID
            )
        results_text = gx.results_text
        for cit, bbox in gx.citations:
            citations.append(ExtractedField(value=cit, bbox=bbox, confidence=CONF_GROBID))
    except Exception as e:
        errors.append(f"grobid: {e}")

    # 3. Papers With Code
    tools_code: list[ExtractedField[str]] = []
    try:
        url = pwc.github_url_for(arxiv_id)
        if url:
            tools_code.append(ExtractedField(value=url, bbox=None, confidence=CONF_PWC))
    except Exception as e:
        errors.append(f"pwc: {e}")

    # 4. key_results regex (with conclusion section as fallback)
    conclusion_text = None
    if "gx" in locals():
        conclusion_text = gx.conclusion_text  # type: ignore[has-type]
    krs: list[ExtractedField[str]] = []
    for kr in key_results.extract(results_text, fallback_text=conclusion_text):
        krs.append(ExtractedField(value=kr, bbox=None, confidence=CONF_REGEX))

    paper = Paper(
        title=title,
        authors=authors,
        abstract=abstract,
        methods=methods,
        datasets=[],
        tools_code=tools_code,
        key_results=krs,
        citations=citations,
    )
    return BuildResult(paper=paper, arxiv_id=arxiv_id, errors=errors)
