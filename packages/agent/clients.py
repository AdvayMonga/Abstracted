"""Async manager for all five MCP servers.

Spawns each as a stdio subprocess, exposes a single `call(server, tool, args)`
entry point. Use as an async context manager:

    async with MCPClients() as clients:
        result = await clients.call("arxiv_fetch", "fetch_metadata", {"arxiv_id": "..."})
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVERS: dict[str, list[str]] = {
    "arxiv_fetch": ["uv", "run", "python", "-m", "packages.tools.arxiv_fetch"],
    "github_lookup": ["uv", "run", "python", "-m", "packages.tools.github_lookup"],
    "semantic_scholar": ["uv", "run", "python", "-m", "packages.tools.semantic_scholar"],
    "obsidian_vault": ["uv", "run", "python", "-m", "packages.tools.obsidian_vault"],
    "review_queue": ["uv", "run", "python", "-m", "packages.tools.review_queue"],
}


def _parse_result(result: Any) -> Any:
    """Extract the structured payload from an MCP CallToolResult."""
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        # FastMCP wraps scalar/list returns under {"result": ...}.
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    for block in result.content:
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return text
    return None


class MCPClients:
    def __init__(self, servers: dict[str, list[str]] | None = None) -> None:
        self._cfg = servers or SERVERS
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}

    async def __aenter__(self) -> MCPClients:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for name, cmd in self._cfg.items():
            params = StdioServerParameters(command=cmd[0], args=cmd[1:])
            r, w = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(r, w))
            await session.initialize()
            self._sessions[name] = session
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)
        self._stack = None
        self._sessions = {}

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        if server not in self._sessions:
            raise KeyError(f"No MCP session for {server!r}. Known: {sorted(self._sessions)}")
        result = await self._sessions[server].call_tool(tool, args)
        if result.isError:
            raise RuntimeError(f"MCP {server}.{tool} failed: {result.content!r}")
        return _parse_result(result)
