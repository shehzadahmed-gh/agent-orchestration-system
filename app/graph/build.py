"""Graph wiring.

                                  +--> math_agent -----+
    retrieve_memory -> supervisor +--> research_agent -+--> finalize -> END
                                  +--> general_agent ---+       ^
                                  +--> human_escalation --------+
                                            |  (interrupt, waits for a human)
                                            v
                                     approved -> general_agent
                                     otherwise -> finalize

retrieve_memory reads similar past interactions from Chroma before routing,
and finalize writes the new one back after the answer is produced.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    human_escalation_node,
    make_finalize_node,
    make_general_agent_node,
    make_math_agent_node,
    make_research_agent_node,
    make_retrieve_memory_node,
    make_supervisor_node,
)
from app.graph.state import AgentState
from app.llm_provider import LLMProvider
from app.memory.semantic_memory import SemanticMemory


def route_after_supervisor(state: AgentState) -> str:
    return state.get("route") or "general"


def route_after_escalation(state: AgentState) -> str:
    response = (state.get("human_response") or "").strip().lower()
    return "approved" if response.startswith("approve") else "finalize"


def build_graph(llm: LLMProvider, memory: SemanticMemory) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_memory", make_retrieve_memory_node(memory))
    graph.add_node("supervisor", make_supervisor_node(llm))
    graph.add_node("math_agent", make_math_agent_node(llm))
    graph.add_node("research_agent", make_research_agent_node(llm))
    graph.add_node("general_agent", make_general_agent_node(llm))
    graph.add_node("human_escalation", human_escalation_node)
    graph.add_node("finalize", make_finalize_node(memory))

    graph.add_edge(START, "retrieve_memory")
    graph.add_edge("retrieve_memory", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "math": "math_agent",
            "research": "research_agent",
            "general": "general_agent",
            "escalate": "human_escalation",
        },
    )
    graph.add_conditional_edges(
        "human_escalation",
        route_after_escalation,
        {"approved": "general_agent", "finalize": "finalize"},
    )
    graph.add_edge("math_agent", "finalize")
    graph.add_edge("research_agent", "finalize")
    graph.add_edge("general_agent", "finalize")
    graph.add_edge("finalize", END)

    return graph
