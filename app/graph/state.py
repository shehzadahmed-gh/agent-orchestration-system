from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task_id: str
    route: str | None  # "math" | "research" | "general" | "escalate", set by the supervisor
    memory_context: list[str]  # snippets pulled from semantic memory before routing
    needs_human_review: bool
    escalation_question: str | None
    human_response: str | None
    final_answer: str | None
