"""Run the review_queue MCP server. `python -m packages.tools.review_queue`."""

from dotenv import load_dotenv

from packages.tools.review_queue.server import mcp

load_dotenv()

if __name__ == "__main__":
    mcp.run()
