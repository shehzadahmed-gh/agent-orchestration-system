"""Reasoning backends for the graph's nodes.

Every node calls one of two methods: `classify_route` (used by the supervisor)
or `respond` (used by the agents). The OpenAI and Anthropic providers build a
prompt and call the model through `generate`. `LocalRuleBasedProvider` uses
keyword rules instead, so the whole graph can run and be tested without an
API key. Select one with LLM_PROVIDER.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

DEFAULT_ESCALATION_KEYWORDS = "delete,wire transfer,payment,refund,cancel subscription,irreversible"


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        raise NotImplementedError

    def classify_route(self, user_message: str, memory_context: list[str]) -> str:
        system = (
            "Classify the user's request into exactly one label: math, research, general, or escalate. "
            "Use 'escalate' only if the request involves a sensitive or irreversible real-world action "
            "(payments, deletions, cancellations, transfers, etc). Respond with only the single label word."
        )
        user = f"Request: {user_message}"
        if memory_context:
            user += "\nRelevant memory:\n" + "\n".join(f"- {m}" for m in memory_context)
        raw = self.generate(system=system, user=user, max_tokens=10).strip().lower()
        for label in ("escalate", "math", "research", "general"):
            if label in raw:
                return label
        return "general"

    def respond(self, question: str, context: list[str] | None = None, tool_output: str | None = None) -> str:
        parts = [f"Question: {question}"]
        if context:
            parts.append("Relevant memory:\n" + "\n".join(f"- {c}" for c in context))
        if tool_output:
            parts.append(f"Tool result:\n{tool_output}")
        return self.generate(
            system=(
                "You are a helpful assistant. Answer clearly and concisely, "
                "using any tool result and memory context provided."
            ),
            user="\n\n".join(parts),
            max_tokens=300,
        )


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def _keyword_pattern(keywords: str) -> re.Pattern[str]:
    """Match any keyword at the start of a word, so "delete" matches "deleted"
    and "deletes" but not "undelete". Multi-word keywords allow any whitespace."""
    parts = [k.strip() for k in keywords.split(",") if k.strip()]
    alternatives = [r"\s+".join(re.escape(word) for word in part.split()) for part in parts]
    return re.compile(r"\b(?:" + "|".join(alternatives) + ")", re.IGNORECASE)


class LocalRuleBasedProvider(LLMProvider):
    """Keyword and regex routing with tool output passed straight through.

    This exercises the graph's control flow without an API key. It has no
    language understanding: "What does delete do in SQL?" is escalated because
    it contains "delete".
    """

    name = "local-rules"

    # A digit on both sides of the operator, so "90-minute" or "step-3" don't
    # route to the math agent.
    _MATH_PATTERN = re.compile(r"\d\s*[-+*/^]\s*\d|\bcalculate\b|\bcompute\b")
    _RESEARCH_TRIGGERS = ("who", "what", "when", "where", "why", "how", "find", "search", "look up", "explain")

    def __init__(self, escalation_keywords: str = DEFAULT_ESCALATION_KEYWORDS):
        self._escalation_pattern = _keyword_pattern(escalation_keywords)

    def generate(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        return "[local-rules] set LLM_PROVIDER=openai or anthropic for generated text."

    def classify_route(self, user_message: str, memory_context: list[str]) -> str:
        lowered = user_message.lower()
        if self._escalation_pattern.search(lowered):
            return "escalate"
        if self._MATH_PATTERN.search(lowered):
            return "math"
        if any(lowered.startswith(t) or f" {t} " in f" {lowered} " for t in self._RESEARCH_TRIGGERS):
            return "research"
        return "general"

    def respond(self, question: str, context: list[str] | None = None, tool_output: str | None = None) -> str:
        if tool_output:
            return tool_output
        if context:
            return "Based on a similar past interaction: " + _clean_memory_snippet(context[0])
        return f"Noted: {question!r}. (Set LLM_PROVIDER=openai or anthropic for a generated response.)"


def _clean_memory_snippet(text: str, max_len: int = 150) -> str:
    """Memory entries are stored as 'Q: ...\\nA: ...'. Return the answer half,
    truncated, so repeated turns don't keep nesting earlier answers."""
    if "\nA: " in text:
        text = text.split("\nA: ", 1)[1]
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def build_llm_provider(settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set.")
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_chat_model)
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set.")
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=settings.anthropic_chat_model)
    return LocalRuleBasedProvider(escalation_keywords=settings.escalation_keywords)
