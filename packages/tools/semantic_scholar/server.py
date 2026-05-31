"""semantic_scholar MCP server: look up papers and find related work via S2."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from packages.shared import semantic_scholar

mcp = FastMCP("semantic_scholar")


@mcp.tool()
def lookup(arxiv_id: str) -> dict | None:
    """Look up paper metadata by arXiv id. Returns None if not indexed.

    Fields: paperId, title, year, authors, citationCount, abstract, venue,
    externalIds.

    Args:
        arxiv_id: arXiv identifier, e.g. "2506.11419".
    """
    return semantic_scholar.lookup(arxiv_id)


@mcp.tool()
def recommendations(paper_id: str, limit: int = 10) -> list[dict]:
    """Return Semantic-Scholar-recommended related papers.

    Args:
        paper_id: an S2 paperId, or "arXiv:<id>" / "DOI:<doi>" form.
        limit: max number of recommendations (default 10).
    """
    return semantic_scholar.recommendations(paper_id, limit=limit)
