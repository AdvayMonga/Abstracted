"""github_lookup MCP server.

Two ways to find code for a paper:
  scan_pdf  — regex over the PDF text (works for any paper, no network).
  pwc_lookup — Papers With Code API (curated, but spotty on recent papers).

The agent can call both and merge; the canonical official repo usually shows
up in scan_pdf for recent papers and pwc_lookup for older ones.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from packages.shared.groundtruth import github_links, pwc

mcp = FastMCP("github_lookup")


@mcp.tool()
def scan_pdf(pdf_path: str) -> list[str]:
    """Scan a PDF for github.com/<user>/<repo> URLs. Returns deduped, canonicalized list.

    Args:
        pdf_path: absolute or repo-relative path to a PDF on disk.
    """
    return github_links.scan(Path(pdf_path))


@mcp.tool()
def pwc_lookup(arxiv_id: str) -> str | None:
    """Look up the official GitHub repo for an arxiv paper via Papers With Code.

    Returns the URL, or None if PwC has no record. Returns the official repo
    if flagged, else the first listed.

    Args:
        arxiv_id: arXiv identifier, e.g. "2506.11419".
    """
    return pwc.github_url_for(arxiv_id)
