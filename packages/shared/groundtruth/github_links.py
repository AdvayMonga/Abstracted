"""Scan a PDF for github.com URLs. Dedupe, drop obvious junk."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pymupdf  # type: ignore[import-untyped]

# user/repo. Trim trailing punctuation that comes from PDF line-wraps and footnotes.
_GH = re.compile(
    r"https?://(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)/"
    r"([A-Za-z0-9._-]+?)(?=[\s)\],;.\"'>]|$)",
    re.IGNORECASE,
)

# repos like "actions", "issues", "blog" are not user repos
_USER_BLOCKLIST = {"about", "explore", "marketplace", "pricing", "topics", "settings"}


def scan(pdf_path: Path, max_pages: int | None = None) -> list[str]:
    """Return canonical https://github.com/<user>/<repo> URLs found anywhere in the PDF."""
    found: list[str] = []
    seen: set[str] = set()
    with pymupdf.open(pdf_path) as doc:
        n = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        # join words instead of lines to defeat PDF line-breaks inside URLs
        for i in range(n):
            text = cast(str, doc[i].get_text("text"))
            # collapse line-wraps that PDF text extraction inserts mid-URL
            joined = re.sub(r"-?\s*\n\s*", "", text)
            for m in _GH.finditer(joined):
                user, repo = m.group(1), m.group(2).rstrip(".")
                if user.lower() in _USER_BLOCKLIST:
                    continue
                url = f"https://github.com/{user}/{repo}"
                if url not in seen:
                    seen.add(url)
                    found.append(url)
    return found
