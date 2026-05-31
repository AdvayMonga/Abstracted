"""Live smoke test: spawn the server over stdio, call fetch_metadata via MCP.

Run with: `uv run python -m packages.tools.arxiv_fetch.smoke <arxiv_id>`.
"""

from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _run(arxiv_id: str) -> None:
    params = StdioServerParameters(
        command="uv", args=["run", "python", "-m", "packages.tools.arxiv_fetch"]
    )
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = await s.list_tools()
        print("tools:", [t.name for t in tools.tools])
        result = await s.call_tool("fetch_metadata", {"arxiv_id": arxiv_id})
        print("fetch_metadata result:")
        for c in result.content:
            text = getattr(c, "text", None)
            if text:
                print(text)


def main() -> None:
    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2506.11419"
    asyncio.run(_run(arxiv_id))


if __name__ == "__main__":
    main()
