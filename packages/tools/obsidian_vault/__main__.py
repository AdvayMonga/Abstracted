"""Run the obsidian_vault MCP server. `python -m packages.tools.obsidian_vault`."""

from packages.tools.obsidian_vault.server import mcp

if __name__ == "__main__":
    mcp.run()
