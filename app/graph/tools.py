"""
Tools available to the agent graph. Both are real, working implementations
(not stubs):

  * `calculator` evaluates arithmetic via a restricted AST walker - safe by
    construction, no `eval()`.
  * `knowledge_base_search` calls the *rag-hybrid-search* project's
    `/retrieval/search` endpoint over HTTP when RAG_SERVICE_URL is
    configured, letting these two portfolio projects compose: the agent
    system can use the RAG pipeline as a tool. It degrades gracefully (a
    clear message, not an error) when that service isn't configured/running.
"""
from __future__ import annotations

import httpx
from langchain_core.tools import tool

from app.config import get_settings
from app.graph.safe_math import UnsafeExpressionError, safe_calculate


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4) / 2'.
    Supports + - * / // % ** and parentheses. Not a general-purpose code interpreter."""
    try:
        result = safe_calculate(expression)
    except UnsafeExpressionError as exc:
        return f"Could not evaluate '{expression}': {exc}"
    return f"{expression.strip()} = {result}"


@tool
def knowledge_base_search(query: str) -> str:
    """Search the internal knowledge base for information relevant to `query`.
    Backed by the rag-hybrid-search hybrid-retrieval service."""
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
    lines = [f"- ({r['source']}) {r['text'][:300]}" for r in results]
    return "\n".join(lines)


ALL_TOOLS = [calculator, knowledge_base_search]
