"""Tests for the GitHub URL scanner."""

from __future__ import annotations

import pymupdf  # type: ignore[import-untyped]

from packages.shared.groundtruth.github_links import scan


def _make_pdf(tmp_path, text: str):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=9)
    out = tmp_path / "p.pdf"
    doc.save(out)
    doc.close()
    return out


def test_finds_simple_url(tmp_path) -> None:
    p = _make_pdf(tmp_path, "Code at https://github.com/foo/bar for details.")
    assert scan(p) == ["https://github.com/foo/bar"]


def test_dedupes_repeated(tmp_path) -> None:
    p = _make_pdf(
        tmp_path,
        "See https://github.com/foo/bar and also https://github.com/foo/bar again",
    )
    assert scan(p) == ["https://github.com/foo/bar"]


def test_strips_trailing_punct(tmp_path) -> None:
    p = _make_pdf(tmp_path, "Available at https://github.com/foo/bar.")
    assert scan(p) == ["https://github.com/foo/bar"]


def test_ignores_non_repo_paths(tmp_path) -> None:
    p = _make_pdf(tmp_path, "Visit https://github.com/about for info")
    assert scan(p) == []
