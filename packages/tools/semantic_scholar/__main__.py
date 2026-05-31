"""Run the semantic_scholar MCP server. `python -m packages.tools.semantic_scholar`."""

from packages.tools.semantic_scholar.server import mcp

if __name__ == "__main__":
    mcp.run()
