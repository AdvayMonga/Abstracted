"""Thin Semantic Scholar Graph API client.

No key needed at low rates. Set SEMANTIC_SCHOLAR_API_KEY to raise the limit.
Keep this small — the agent only needs "look up this paper" and "recommend
related papers".
"""

from __future__ import annotations

import os
import time

import httpx

GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
REC_BASE = "https://api.semanticscholar.org/recommendations/v1"
USER_AGENT = "abstracted-research-agent/0.1 (https://github.com/local; contact via repo)"
TIMEOUT = 20.0
RETRY_DELAY = 2.0
MAX_RETRIES = 3


def _client() -> httpx.Client:
    headers = {"User-Agent": USER_AGENT}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        headers["x-api-key"] = key
    return httpx.Client(timeout=TIMEOUT, headers=headers)


def _get(client: httpx.Client, url: str, params: dict[str, str]) -> httpx.Response:
    delay = RETRY_DELAY
    for _ in range(MAX_RETRIES):
        r = client.get(url, params=params)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(delay)
            delay *= 2
            continue
        return r
    return r


def lookup(arxiv_id: str) -> dict | None:
    """Return basic paper info keyed by arxiv id, or None if not indexed."""
    fields = "paperId,title,year,authors,citationCount,abstract,venue,externalIds"
    with _client() as c:
        r = _get(c, f"{GRAPH_BASE}/paper/arXiv:{arxiv_id}", {"fields": fields})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()


def recommendations(paper_id: str, limit: int = 10) -> list[dict]:
    """Return up to `limit` Semantic-Scholar-recommended related papers.

    paper_id can be an S2 id, "arXiv:<id>", "DOI:<doi>", etc.
    """
    fields = "paperId,title,year,authors,citationCount,externalIds"
    with _client() as c:
        r = _get(
            c,
            f"{REC_BASE}/papers/forpaper/{paper_id}",
            {"fields": fields, "limit": str(limit)},
        )
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json().get("recommendedPapers") or []
