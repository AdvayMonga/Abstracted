"""arxiv_fetch MCP server.

Wraps packages.shared.groundtruth.arxiv as MCP tools so the agent can call
arXiv lookups without owning the HTTP layer.

Tools:
  fetch_metadata(arxiv_id) -> dict
  download_pdf(arxiv_id, dest_dir) -> str (path)
  search(categories, max_results) -> list[str]
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from packages.shared.groundtruth import arxiv

mcp = FastMCP("arxiv_fetch")


@mcp.tool()
def fetch_metadata(arxiv_id: str) -> dict:
    """Fetch title, authors, abstract, and categories for an arXiv paper.

    Args:
        arxiv_id: arXiv identifier, e.g. "2506.11419" or "cs.LG/0701001".
    """
    return asdict(arxiv.fetch_metadata(arxiv_id))


@mcp.tool()
def download_pdf(arxiv_id: str, dest_dir: str = "data/raw") -> str:
    """Download the PDF for an arXiv paper. Returns the path written.

    Args:
        arxiv_id: arXiv identifier.
        dest_dir: directory to write the PDF into. Created if missing.
    """
    return str(arxiv.download_pdf(arxiv_id, Path(dest_dir)))


@mcp.tool()
def search(categories: list[str], max_results: int = 20) -> list[str]:
    """Return arXiv IDs matching any of the given primary categories.

    Args:
        categories: list of arXiv category codes, e.g. ["cs.LG", "cs.CL"].
        max_results: cap on results.
    """
    return arxiv.search(categories, max_results=max_results)
