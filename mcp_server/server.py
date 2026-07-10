"""MCP server exposing tools for the Research Analyst Agent.

Phase 1: only `web_search` is implemented. `read_document` and
`query_vector_db` are added in Phase 2.

Runs over stdio transport (recommended for local dev per the project spec).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# Load environment variables from .env for runtime configuration.
from dotenv import load_dotenv
# FastMCP is the MCP server implementation used to expose tools.
from mcp.server.fastmcp import FastMCP

# Web search implementation and error type imported from tools.
from tools.web_search import WebSearchError
from tools.web_search import web_search as run_web_search

# Determine the repository root path relative to this file.
REPO_ROOT = Path(__file__).resolve().parent.parent
# Load environment settings from the repository .env file.
load_dotenv(REPO_ROOT / ".env")

# Set up the log file path for MCP tool calls.
LOG_PATH = REPO_ROOT / "logs" / "mcp_server.log"
# Create the log directory if it doesn't already exist.
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Configure the root logger to print INFO-level events.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

# Create a FastMCP server instance with the service name.
mcp = FastMCP("research-analyst-mcp")


def log_tool_call(tool_name: str, tool_input: dict, tool_output) -> None:
    # Build a log entry with timestamp, tool name, input, and output.
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "input": tool_input,
        "output": tool_output,
    }
    # Append the log entry as a JSON line to the MCP server log file.
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# Register web_search as an MCP tool callable by clients.
@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for current information relevant to a query.

    Use this to find recent articles, reports, or job postings about the
    tech job market that aren't in the local knowledge base.

    Args:
        query: the search query (e.g. "AI engineer salary trends 2026")
        max_results: maximum number of results to return (default 5)
    """
    # Collect the input values so they can be logged later.
    tool_input = {"query": query, "max_results": max_results}

    try:
        # Execute the actual search implementation from tools.web_search.
        results = run_web_search(query, max_results)
    except WebSearchError as exc:
        # If the search fails, return an error structure and log the failure.
        results = {"error": str(exc)}
        logger.error("web_search failed: %s", exc)

    # Log every tool call for observability and debugging.
    log_tool_call("web_search", tool_input, results)
    # Return the search results to the MCP client.
    return results


if __name__ == "__main__":
    # Start the MCP server using stdio transport when run as a script.
    mcp.run(transport="stdio")
