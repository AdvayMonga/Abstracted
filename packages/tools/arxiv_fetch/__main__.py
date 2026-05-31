"""Run the arxiv_fetch MCP server over stdio. `python -m packages.tools.arxiv_fetch`."""

from packages.tools.arxiv_fetch.server import mcp

if __name__ == "__main__":
    mcp.run()
