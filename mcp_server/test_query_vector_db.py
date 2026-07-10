"""Standalone smoke test for the query_vector_db MCP tool (Phase 2).

Spawns server.py over stdio, calls query_vector_db with a query that should
clearly match job-posting-ai-engineer-novagrid.md, and confirms that doc_id
shows up in the results.

Usage:
    source .venv/bin/activate
    python mcp_server/test_query_vector_db.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"
EXPECTED_DOC_ID = "job-posting-ai-engineer-novagrid"


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            assert any(t.name == "query_vector_db" for t in tools_response.tools), (
                "query_vector_db tool was not exposed by the server"
            )
            tool = next(t for t in tools_response.tools if t.name == "query_vector_db")
            print(f"Discovered tool: {tool.name}")
            print(f"  input_schema: {json.dumps(tool.inputSchema)}")
            schema_props = tool.inputSchema.get("properties", {})
            assert "query" in schema_props, "query_vector_db schema missing 'query' param"
            assert "top_k" in schema_props, "query_vector_db schema missing 'top_k' param"
            print("Schema check passed: query_vector_db has 'query' and 'top_k'.")

            query = "salary range and required skills for an AI Engineer role at NovaGrid"
            print(f"\nCalling query_vector_db(query={query!r}, top_k=3)...")
            result = await session.call_tool("query_vector_db", {"query": query, "top_k": 3})

            print("\nRaw tool result content:")
            matched_doc_ids = []
            for block in result.content:
                text = getattr(block, "text", block)
                print(text)
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and "source_doc_id" in parsed:
                        matched_doc_ids.append(parsed["source_doc_id"])
                except (json.JSONDecodeError, TypeError):
                    pass

            assert EXPECTED_DOC_ID in matched_doc_ids, (
                f"Expected '{EXPECTED_DOC_ID}' in top results, got: {matched_doc_ids}"
            )
            print(f"\nConfirmed '{EXPECTED_DOC_ID}' appears in the top results.")
            print("\nPhase 2 query_vector_db smoke test: OK")


if __name__ == "__main__":
    asyncio.run(main())
