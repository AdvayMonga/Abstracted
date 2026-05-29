"""Tests for arXiv metadata parsing. HTTP is mocked."""

from __future__ import annotations

import httpx
import pytest

from packages.shared.groundtruth import arxiv

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on complex
    recurrent or convolutional neural networks.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <category xmlns="http://arxiv.org/schemas/atom" term="cs.CL"/>
    <category xmlns="http://arxiv.org/schemas/atom" term="cs.LG"/>
  </entry>
</feed>
"""


def _mock_client(text: str, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=text)

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_fetch_metadata_parses_atom() -> None:
    client = _mock_client(SAMPLE_ATOM)
    meta = arxiv.fetch_metadata("1706.03762", client=client)
    assert meta.title == "Attention Is All You Need"
    assert meta.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert "sequence transduction" in meta.abstract
    assert "cs.CL" in meta.categories and "cs.LG" in meta.categories


def test_fetch_metadata_rejects_bad_id() -> None:
    with pytest.raises(ValueError):
        arxiv.fetch_metadata("not-an-id")


def test_search_extracts_ids() -> None:
    feed = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/abs/2403.00001v2</id></entry>
      <entry><id>http://arxiv.org/abs/2403.00002v1</id></entry>
    </feed>
    """
    client = _mock_client(feed)
    ids = arxiv.search(["cs.LG"], max_results=2, client=client)
    assert ids == ["2403.00001", "2403.00002"]
