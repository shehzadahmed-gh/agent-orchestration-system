from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from app.graph.state import AgentState
from app.graph.tools import calculator, knowledge_base_search
from app.llm_provider import LLMProvider
from app.memory.semantic_memory import SemanticMemory

_EXPR_ALLOWED_CHARS = set("0123456789+-*/().% \t")


def _last_human_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _extract_expression(text: str) -> str | None:
    normalized = text.replace("^", "**")
    filtered = "".join(ch if ch in _EXPR_ALLOWED_CHARS else " " for ch in normalized)
    cleaned = re.sub(r"\s+", " ", filtered).strip()
    return cleaned if any(ch.isdigit() for ch in cleaned) else None


def make_retrieve_memory_node(memory: SemanticMemory):
    def retrieve_memory_node(state: AgentState) -> dict:
        query = _last_human_text(state["messages"])
        hits = memory.search(query, top_k=3) if query else []
        return {"memory_context": hits}

    return retrieve_memory_node


def make_supervisor_node(llm: LLMProvider):
    def supervisor_node(state: AgentState) -> dict:
        user_msg = _last_human_text(state["messages"])
        route = llm.classify_route(user_msg, state.get("memory_context", []))
        updates: dict = {"route": route}
        if route == "escalate":
            updates["needs_human_review"] = True
            updates["escalation_question"] = (
                f"This request was flagged as needing human approval before proceeding: {user_msg!r}. "
                "Reply with 'approve', 'reject', or free-text guidance."
            )
        return updates

    return supervisor_node


def make_math_agent_node(llm: LLMProvider):
    def math_agent_node(state: AgentState) -> dict:
        user_msg = _last_human_text(state["messages"])
        expression = _extract_expression(user_msg)
        if expression:
            tool_output = calculator.invoke({"expression": expression})
            if tool_output.startswith("Could not evaluate"):
                tool_output = (
                    f"This looked like it might contain a calculation, but I couldn't parse a valid "
                    f"arithmetic expression from it (tried: '{expression}'). Try phrasing it as a plain "
                    f"expression, e.g. '12 * (3 + 4) / 2'."
                )
        else:
            tool_output = "No arithmetic expression was found in the request."
        answer = llm.respond(user_msg, tool_output=tool_output)
        return {"messages": [AIMessage(content=answer)], "final_answer": answer}

    return math_agent_node


def make_research_agent_node(llm: LLMProvider):
    def research_agent_node(state: AgentState) -> dict:
        user_msg = _last_human_text(state["messages"])
        tool_output = knowledge_base_search.invoke({"query": user_msg})
        answer = llm.respond(user_msg, context=state.get("memory_context"), tool_output=tool_output)
        return {"messages": [AIMessage(content=answer)], "final_answer": answer}

    return research_agent_node


def make_general_agent_node(llm: LLMProvider):
    def general_agent_node(state: AgentState) -> dict:
        user_msg = _last_human_text(state["messages"])
        answer = llm.respond(user_msg, context=state.get("memory_context"))
        return {"messages": [AIMessage(content=answer)], "final_answer": answer}

    return general_agent_node


def human_escalation_node(state: AgentState) -> dict:
    """
    Pauses the graph via LangGraph's interrupt() and waits for a human to
    call POST /escalations/{id}/respond. LangGraph re-runs a node from the
    top on resume, so everything before interrupt() here must stay
    idempotent - which is why this node does nothing but read state and
    interrupt: no writes happen until after the human's response comes back.
    """
    question = state.get("escalation_question") or "This request needs human approval. Approve, reject, or advise?"
    response = interrupt({"question": question, "route": state.get("route")})
    return {"human_response": str(response)}


def make_finalize_node(memory: SemanticMemory):
    def finalize_node(state: AgentState) -> dict:
        user_msg = _last_human_text(state["messages"])
        final = state.get("final_answer")
        if not final:
            human_resp = state.get("human_response")
            final = (
                f"The request was not approved for automated handling. Human response: {human_resp}"
                if human_resp is not None
                else "No answer was produced."
            )

        memory.add(
            text=f"Q: {user_msg}\nA: {final}",
            metadata={"task_id": state["task_id"], "route": state.get("route") or "unknown"},
        )
        return {"final_answer": final, "messages": [AIMessage(content=final)]}

    return finalize_node
