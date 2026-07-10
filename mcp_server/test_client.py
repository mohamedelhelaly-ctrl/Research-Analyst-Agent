"""Standalone smoke test for the MCP server (Phase 1).

Spawns server.py as a subprocess over stdio, lists the tools it exposes,
verifies the web_search tool schema, and makes one real tool call.

Usage:
    source mcp_server/.venv/bin/activate
    python mcp_server/test_client.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Locate server.py relative to this test file.
SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"


async def main() -> None:
    # Build the parameters needed to start the MCP server over stdio.
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)])

    # Launch the server process and obtain async read/write streams.
    async with stdio_client(params) as (read, write):
        # Create an MCP client session over the stdio streams.
        async with ClientSession(read, write) as session:
            # Send the initialize request to start the MCP protocol handshake.
            await session.initialize()

            # Request the list of tools exposed by the server.
            tools_response = await session.list_tools()
            print("Discovered tools:")

            # Print each tool name, description, and its input schema.
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description!r}")
                print(f"    input_schema: {json.dumps(tool.inputSchema)}")

            # Ensure the web_search tool is present in the tool list.
            assert any(t.name == "web_search" for t in tools_response.tools), (
                "web_search tool was not exposed by the server"
            )

            # Select the web_search tool object from the tool list.
            web_search_tool = next(t for t in tools_response.tools if t.name == "web_search")

            # Extract the schema properties for the web_search tool.
            schema_props = web_search_tool.inputSchema.get("properties", {})

            # Validate that the schema defines required parameters.
            assert "query" in schema_props, "web_search schema missing 'query' param"
            assert "max_results" in schema_props, "web_search schema missing 'max_results' param"
            print("\nSchema check passed: web_search has 'query' and 'max_results'.")

            # Call the web_search tool with a sample query.
            print("\nCalling web_search(query='AI engineer job market trends 2026', max_results=3)...")
            result = await session.call_tool(
                "web_search",
                {"query": "AI engineer job market trends 2026", "max_results": 3},
            )

            # Print each block of content returned by the tool call.
            print("\nRaw tool result content:")
            for block in result.content:
                # Some tool results may store output in a 'text' attribute.
                text = getattr(block, "text", block)
                print(text)

            # Print a final confirmation message when the smoke test passes.
            print("\nPhase 1 smoke test: OK")


if __name__ == "__main__":
    # Run the async main function when this file is executed directly.
    asyncio.run(main())
