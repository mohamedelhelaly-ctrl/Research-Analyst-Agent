"""Standalone smoke test for the read_document MCP tool (Phase 2).

Spawns server.py over stdio, reads a known doc_id and confirms the full
text comes back, then confirms an unknown doc_id is reported as an error
rather than crashing the server.

Usage:
    source .venv/bin/activate
    python mcp_server/test_read_document.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"
KNOWN_DOC_ID = "job-posting-ai-engineer-novagrid"
UNKNOWN_DOC_ID = "does-not-exist"


def _extract_text(result) -> str:
    parts = []
    for block in result.content:
        parts.append(getattr(block, "text", str(block)))
    return "\n".join(parts)


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            assert any(t.name == "read_document" for t in tools_response.tools), (
                "read_document tool was not exposed by the server"
            )
            tool = next(t for t in tools_response.tools if t.name == "read_document")
            print(f"Discovered tool: {tool.name}")
            print(f"  input_schema: {json.dumps(tool.inputSchema)}")
            assert "doc_id" in tool.inputSchema.get("properties", {}), (
                "read_document schema missing 'doc_id' param"
            )
            print("Schema check passed: read_document has 'doc_id'.")

            print(f"\nCalling read_document(doc_id={KNOWN_DOC_ID!r})...")
            result = await session.call_tool("read_document", {"doc_id": KNOWN_DOC_ID})
            text = _extract_text(result)
            print(text)
            assert "NovaGrid AI" in text, "Expected doc content not found in read_document output"
            assert "$145,000" in text, "Expected specific fact not found in read_document output"
            print(f"\nConfirmed full text of '{KNOWN_DOC_ID}' was returned with expected facts.")

            print(f"\nCalling read_document(doc_id={UNKNOWN_DOC_ID!r}) (expect graceful error)...")
            error_result = await session.call_tool("read_document", {"doc_id": UNKNOWN_DOC_ID})
            error_text = _extract_text(error_result)
            print(error_text)
            assert "error" in error_text.lower(), "Expected an error message for unknown doc_id"
            print("Confirmed unknown doc_id is reported as an error, not a crash.")

            print("\nPhase 2 read_document smoke test: OK")


if __name__ == "__main__":
    asyncio.run(main())
