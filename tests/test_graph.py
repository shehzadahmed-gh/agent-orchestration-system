"""Runs the compiled graph in-process with an in-memory checkpointer, so the
routing, tool calls and interrupt/resume path are tested without Postgres,
Redis or a Celery worker."""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.embeddings import LocalHashingEmbeddingProvider
from app.graph.build import build_graph
from app.llm_provider import LocalRuleBasedProvider
from app.memory.semantic_memory import SemanticMemory


@pytest.fixture()
def graph(tmp_path):
    memory = SemanticMemory(
        embedder=LocalHashingEmbeddingProvider(), collection_name="graph_test", local_path=str(tmp_path)
    )
    return build_graph(LocalRuleBasedProvider(), memory).compile(checkpointer=MemorySaver())


def _run(graph, text, thread_id):
    state = {
        "messages": [HumanMessage(content=text)],
        "task_id": thread_id,
        "route": None,
        "memory_context": [],
        "needs_human_review": False,
        "escalation_question": None,
        "human_response": None,
        "final_answer": None,
    }
    return graph.invoke(state, {"configurable": {"thread_id": thread_id}})


def test_math_request_uses_calculator(graph):
    result = _run(graph, "What is 12 * (3 + 4) / 2?", "t1")
    assert result["route"] == "math"
    assert "42" in result["final_answer"]


@pytest.mark.timeout(10)
def test_huge_power_fails_fast_instead_of_hanging(graph):
    result = _run(graph, "What is 2^9^9^9?", "t2")
    assert result["route"] == "math"
    assert "Could not evaluate" in result["final_answer"]


def test_division_by_zero_returns_a_message(graph):
    result = _run(graph, "What is 5/0?", "t3")
    assert "division by zero" in result["final_answer"]


def test_escalation_pauses_and_approval_resumes(graph):
    config = {"configurable": {"thread_id": "t4"}}
    paused = _run(graph, "Please delete my account.", "t4")
    assert paused["__interrupt__"]
    assert paused.get("final_answer") is None

    resumed = graph.invoke(Command(resume="approve"), config)
    assert resumed["human_response"] == "approve"
    assert resumed["final_answer"]
    assert "not approved" not in resumed["final_answer"]


def test_rejection_finishes_without_running_an_agent(graph):
    config = {"configurable": {"thread_id": "t5"}}
    _run(graph, "Refund my last payment.", "t5")
    resumed = graph.invoke(Command(resume="reject, not authorised"), config)
    assert "not approved" in resumed["final_answer"]
