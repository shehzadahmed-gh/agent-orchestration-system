"""Tools the agents can call.

`knowledge_base_search` calls the /retrieval/search endpoint of the
rag-hybrid-search service (separate repo) when RAG_SERVICE_URL is set. If the
service isn't configured or can't be reached it returns a message instead of
raising, so the research agent still produces an answer.
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from app.config import get_settings
from app.graph.safe_math import CalculationError, safe_calculate


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4) / 2'.
    Supports + - * / // % ** and parentheses."""
    try:
        result = safe_calculate(expression)
    except CalculationError as exc:
        return f"Could not evaluate '{expression}': {exc}"
    return f"{expression.strip()} = {result}"


@tool
def knowledge_base_search(query: str) -> str:
    """Search the internal knowledge base for passages relevant to `query`."""
    settings = get_settings()
    if not settings.rag_service_url:
        return "Knowledge base search is not configured (set RAG_SERVICE_URL to enable it)."
    try:
        response = httpx.post(
            f"{settings.rag_service_url.rstrip('/')}/retrieval/search",
            json={"query": query, "top_k": 3},
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Knowledge base search failed: {exc}"

    results = response.json().get("results", [])
    if not results:
        return "No relevant documents found in the knowledge base."
    return "\n".join(f"- ({r['source']}) {r['text'][:300]}" for r in results)


ALL_TOOLS = [calculator, knowledge_base_search]
