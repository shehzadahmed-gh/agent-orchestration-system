"""
Pluggable reasoning backend for the graph's nodes - the same
offline-default-plus-real-API pattern used throughout both portfolio
projects. Every node calls one of three methods on this interface:

  * classify_route(...)          - used by the supervisor node
  * respond(...)                 - used by every agent node to produce output
  * generate(...)                - the raw primitive real providers implement

`LocalRuleBasedProvider` implements routing/responses with keyword
heuristics and direct tool-output formatting, so the entire graph (routing,
tool use, memory, escalation, resume) is exercisable with zero API keys.
Set LLM_PROVIDER=openai|anthropic for real model-driven reasoning - nothing
else in app/graph changes.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        raise NotImplementedError

    def classify_route(self, user_message: str, memory_context: list[str]) -> str:
        """Default path for real providers: ask the model to classify."""
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
        """Default path for real providers: build a prompt from whatever
        context/tool output is available and ask the model to answer."""
        parts = [f"Question: {question}"]
        if context:
            parts.append("Relevant memory:\n" + "\n".join(f"- {c}" for c in context))
        if tool_output:
            parts.append(f"Tool result:\n{tool_output}")
        return self.generate(
            system="You are a helpful assistant. Answer clearly and concisely, using any tool result and memory context provided.",
            user="\n\n".join(parts),
            max_tokens=300,
        )


class OpenAIProvider(LLMProvider):
    """GPT-4o via the OpenAI Chat Completions API."""

    name = "openai-gpt-4o"

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
    """Claude Sonnet via the Anthropic Messages API."""

    name = "anthropic-claude-sonnet"

    def __init__(self, api_key: str, model: str = "claude-sonnet-5"):
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


class LocalRuleBasedProvider(LLMProvider):
    """
    Deterministic, dependency-free stand-in. Routing uses keyword/regex
    heuristics instead of model reasoning, and responses are built directly
    from tool output / memory rather than generated - real, working logic
    for exercising the whole graph offline, just without genuine language
    understanding. Swap LLM_PROVIDER to unlock real reasoning.
    """

    name = "local-rule-based-stub"

    _ESCALATION_TRIGGERS = ("delete", "wire transfer", "payment", "refund", "cancel subscription", "irreversible")
    # Requires a digit on BOTH sides of the operator, so "90-minute" or
    # "step-3" don't false-positive into the math route the way a looser
    # "digit followed by hyphen" pattern would.
    _MATH_PATTERN = re.compile(r"\d\s*[-+*/^]\s*\d|\bcalculate\b|\bcompute\b")
    _RESEARCH_TRIGGERS = ("who", "what", "when", "where", "why", "how", "find", "search", "look up", "explain")

    def generate(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        return "[local-rule-based-stub] set LLM_PROVIDER=openai or anthropic for real generated text."

    def classify_route(self, user_message: str, memory_context: list[str]) -> str:
        lowered = user_message.lower()
        if any(trigger in lowered for trigger in self._ESCALATION_TRIGGERS):
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
        return f"Noted: {question!r}. (Set LLM_PROVIDER=openai or anthropic for a real generated response.)"


def _clean_memory_snippet(text: str, max_len: int = 150) -> str:
    """Memory entries are stored as 'Q: ...\\nA: ...'; surface just the
    answer half, truncated, so echoing memory in the local stub can't
    compound into ever-longer nested text across repeated turns."""
    if "\nA: " in text:
        text = text.split("\nA: ", 1)[1]
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "\u2026"
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
    return LocalRuleBasedProvider()
