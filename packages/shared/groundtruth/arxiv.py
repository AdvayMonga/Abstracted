"""arXiv API: fetch metadata + PDF for a given arxiv_id."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
QUERY_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{id}.pdf"
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$|^[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(v\d+)?$")
CACHE_DIR = Path("data/cache/arxiv")
FETCH_DELAY = 5.0  # arXiv API politeness (bumped from 3s after rate limit incidents)
MAX_RETRIES = 1  # no retry storms — fail fast and let the caller decide
USER_AGENT = "abstracted-research-agent/0.1 (https://github.com/local; contact via repo)"


def _client(timeout: float = 30.0, follow_redirects: bool = False) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=follow_redirects,
        headers={"User-Agent": USER_AGENT},
    )


def _get_with_backoff(
    client: httpx.Client, url: str, params: dict[str, str] | None = None
) -> httpx.Response:
    """GET with exponential backoff on 429/5xx."""
    delay = FETCH_DELAY
    last_exc: Exception | None = None
    for _ in range(MAX_RETRIES):
        try:
            r = client.get(url, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r
        except httpx.HTTPError as e:
            last_exc = e
            time.sleep(delay)
            delay *= 2
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Exhausted retries for {url}")


@dataclass(frozen=True)
class ArxivMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]


def fetch_metadata(
    arxiv_id: str, client: httpx.Client | None = None, use_cache: bool = True
) -> ArxivMeta:
    if not ARXIV_ID_RE.match(arxiv_id):
        raise ValueError(f"Invalid arxiv id: {arxiv_id!r}")
    if use_cache:
        cache_path = CACHE_DIR / f"{arxiv_id.replace('/', '_')}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            return ArxivMeta(**data)
    own = client is None
    client = client or _client(timeout=30.0)
    try:
        r = _get_with_backoff(client, QUERY_URL, params={"id_list": arxiv_id})
        time.sleep(FETCH_DELAY)
    finally:
        if own:
            client.close()
    root = ET.fromstring(r.text)
    entry = root.find("a:entry", ATOM_NS)
    if entry is None:
        raise LookupError(f"No arXiv entry for {arxiv_id}")
    title_el = entry.find("a:title", ATOM_NS)
    summary_el = entry.find("a:summary", ATOM_NS)
    title = (title_el.text or "").strip() if title_el is not None else ""
    abstract = (summary_el.text or "").strip() if summary_el is not None else ""
    title = re.sub(r"\s+", " ", title)
    abstract = re.sub(r"\s+", " ", abstract)
    authors = []
    for a in entry.findall("a:author/a:name", ATOM_NS):
        if a.text:
            authors.append(a.text.strip())
    cats = []
    # arXiv returns <category term="..."/> in both atom and arxiv-schema namespaces;
    # check both.
    for c in entry.findall("a:category", ATOM_NS):
        term = c.get("term")
        if term:
            cats.append(term)
    for c in entry.findall("{http://arxiv.org/schemas/atom}category"):
        term = c.get("term")
        if term and term not in cats:
            cats.append(term)
    meta = ArxivMeta(arxiv_id, title, authors, abstract, cats)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{arxiv_id.replace('/', '_')}.json"
        cache_path.write_text(json.dumps(asdict(meta)))
    return meta


def download_pdf(arxiv_id: str, dest_dir: Path, client: httpx.Client | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{arxiv_id.replace('/', '_')}.pdf"
    if dest.exists():
        return dest
    own = client is None
    client = client or _client(timeout=60.0, follow_redirects=True)
    try:
        r = _get_with_backoff(client, PDF_URL.format(id=arxiv_id))
        dest.write_bytes(r.content)
    finally:
        if own:
            client.close()
    time.sleep(FETCH_DELAY)
    return dest


def search(
    categories: list[str],
    max_results: int,
    client: httpx.Client | None = None,
) -> list[str]:
    """Search arXiv by category; return list of arxiv_ids (no version)."""
    q = " OR ".join(f"cat:{c}" for c in categories)
    own = client is None
    client = client or _client(timeout=30.0)
    try:
        r = _get_with_backoff(
            client,
            QUERY_URL,
            params={
                "search_query": q,
                "max_results": str(max_results),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        time.sleep(FETCH_DELAY)
    finally:
        if own:
            client.close()
    root = ET.fromstring(r.text)
    ids: list[str] = []
    for entry in root.findall("a:entry", ATOM_NS):
        id_el = entry.find("a:id", ATOM_NS)
        if id_el is None or not id_el.text:
            continue
        # id is a URL like http://arxiv.org/abs/2403.12345v1
        raw = id_el.text.rsplit("/", 1)[-1]
        ids.append(re.sub(r"v\d+$", "", raw))
    return ids
