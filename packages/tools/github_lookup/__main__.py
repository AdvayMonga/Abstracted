"""Run the github_lookup MCP server. `python -m packages.tools.github_lookup`."""

from packages.tools.github_lookup.server import mcp

if __name__ == "__main__":
    mcp.run()
