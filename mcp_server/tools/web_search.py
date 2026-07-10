"""Web search tool backed by Tavily, with a deterministic mock fallback.

The mock fallback lets the MCP server (and anything calling it) run and be
tested end-to-end before a real WEB_SEARCH_API_KEY is available.
"""

import os

from tavily import TavilyClient


class WebSearchError(Exception):
    """Raised when the underlying search API call fails."""


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return a list of {title, url, snippet} results."""
    api_key = os.environ.get("WEB_SEARCH_API_KEY")

    if not api_key:
        return _mock_search(query, max_results)

    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(query=query, max_results=max_results)
    except Exception as exc:
        raise WebSearchError(f"Tavily search failed: {exc}") from exc

    results = []
    for item in response.get("results", [])[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
        )
    return results


def _mock_search(query: str, max_results: int) -> list[dict]:
    return [
        {
            "title": f"Mock result {i + 1} for '{query}'",
            "url": f"https://example.com/mock-result-{i + 1}",
            "snippet": (
                f"Placeholder snippet #{i + 1} for query '{query}'. "
                "Set WEB_SEARCH_API_KEY in .env to get real Tavily results."
            ),
        }
        for i in range(max_results)
    ]
