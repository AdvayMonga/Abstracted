"""Papers With Code: lookup github repo for an arxiv paper.

Uses the public PwC API. Returns None when no repo is linked.
"""

from __future__ import annotations

import httpx

PWC_URL = "https://paperswithcode.com/api/v1/papers/"


def github_url_for(arxiv_id: str, client: httpx.Client | None = None) -> str | None:
    own = client is None
    client = client or httpx.Client(timeout=20.0)
    try:
        # PwC indexes by arxiv id without version.
        r = client.get(f"{PWC_URL}?arxiv_id={arxiv_id}")
        if r.status_code != 200:
            return None
        payload = r.json()
        results = payload.get("results") or []
        if not results:
            return None
        paper_id = results[0]["id"]
        repos = client.get(f"{PWC_URL}{paper_id}/repositories/")
        if repos.status_code != 200:
            return None
        repo_results = repos.json().get("results") or []
        if not repo_results:
            return None
        # Prefer official repo if flagged; else first.
        official = next((r for r in repo_results if r.get("is_official")), None)
        chosen = official or repo_results[0]
        return chosen.get("url")
    finally:
        if own:
            client.close()
