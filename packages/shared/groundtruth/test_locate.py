"""Tests for the PDF bbox locator."""

from __future__ import annotations

import pymupdf  # type: ignore[import-untyped]
import pytest

from packages.shared.groundtruth.locate import locate


@pytest.fixture
def simple_doc() -> pymupdf.Document:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Attention Is All You Need", fontsize=18)
    page.insert_text((72, 160), "Ashish Vaswani, Noam Shazeer", fontsize=11)
    page.insert_text((72, 220), "We propose a simple network architecture", fontsize=10)
    page.insert_text((72, 234), "based solely on attention mechanisms.", fontsize=10)
    return doc


def test_locate_exact_title(simple_doc: pymupdf.Document) -> None:
    bbox = locate(simple_doc, "Attention Is All You Need")
    assert bbox is not None
    assert bbox.page == 0
    assert 70 < bbox.x0 < 120
    # PyMuPDF baseline at y=100; bbox y0 sits at the glyph top, a bit above.
    assert 70 < bbox.y0 < 110


def test_locate_multiline_spans_full_paragraph(simple_doc: pymupdf.Document) -> None:
    query = "We propose a simple network architecture based solely on attention mechanisms"
    bbox = locate(simple_doc, query)
    assert bbox is not None
    # Should span both lines (y0 around 220, y1 around 234).
    assert bbox.y1 - bbox.y0 > 10


def test_locate_returns_none_for_missing(simple_doc: pymupdf.Document) -> None:
    assert locate(simple_doc, "this phrase does not appear anywhere") is None


def test_locate_handles_hyphenation() -> None:
    """A hyphen at line-end should let the locator find the joined word."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "We use convolu-", fontsize=10)
    page.insert_text((72, 114), "tional layers throughout.", fontsize=10)
    bbox = locate(doc, "we use convolutional layers")
    assert bbox is not None
    assert bbox.y1 - bbox.y0 > 5
