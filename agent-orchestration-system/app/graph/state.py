from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task_id: str
    route: Optional[str]  # "math" | "research" | "general" | "escalate", set by the supervisor
    memory_context: list[str]  # snippets pulled from semantic memory before routing
    needs_human_review: bool
    escalation_question: Optional[str]
    human_response: Optional[str]
    final_answer: Optional[str]
