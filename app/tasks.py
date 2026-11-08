"""
The Celery task is where the graph actually executes - decoupled from the
FastAPI request/response cycle, as called for by "distributed background
workloads through Redis and Celery, separating asynchronous processing from
interactive execution". `POST /tasks` returns as soon as this task is
*enqueued*; the graph runs here, in a worker process, and writes its outcome
back to the `tasks` / `escalations` tables that the API reads from.
"""
from __future__ import annotations

from celery.utils.log import get_task_logger
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.celery_app import celery_app
from app.config import get_settings
from app.db.base import SessionLocal
from app.db.models import Escalation, EscalationStatus, Task, TaskStatus
from app.embeddings import build_embedding_provider
from app.graph.build import build_graph
from app.graph.checkpointer import get_checkpointer_ctx
from app.llm_provider import build_llm_provider
from app.memory.semantic_memory import SemanticMemory

logger = get_task_logger(__name__)


def _build_graph_dependencies():
    settings = get_settings()
    llm = build_llm_provider(settings)
    embedder = build_embedding_provider(settings)
    memory = SemanticMemory(
        embedder=embedder,
        collection_name=settings.semantic_memory_collection,
        url=settings.chroma_url,
        local_path=settings.chroma_local_path,
        top_k=settings.semantic_memory_top_k,
    )
    return settings, llm, memory


def _initial_state(task_id: str, input_text: str) -> dict:
    return {
        "messages": [HumanMessage(content=input_text)],
        "task_id": task_id,
        "route": None,
        "memory_context": [],
        "needs_human_review": False,
        "escalation_question": None,
        "human_response": None,
        "final_answer": None,
    }


def _run_or_resume(task_id: str, resume_value: str | None) -> None:
    settings, llm, memory = _build_graph_dependencies()
    db = SessionLocal()
    try:
        task = db.get(Task, task_id)
        if task is None:
            raise ValueError(f"Task {task_id!r} not found")

        task.status = TaskStatus.RUNNING.value
        db.add(task)
        db.commit()

        with get_checkpointer_ctx(settings) as checkpointer:
            compiled = build_graph(llm, memory).compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": task_id}}
            graph_input = Command(resume=resume_value) if resume_value is not None else _initial_state(task_id, task.input_text)
            result = compiled.invoke(graph_input, config)

        interrupts = result.get("__interrupt__")
        if interrupts:
            question = interrupts[0].value.get("question", "This request needs human approval.")
            task.status = TaskStatus.AWAITING_HUMAN.value
            task.route = result.get("route")
            db.add(task)
            db.add(Escalation(task_id=task_id, question=question, status=EscalationStatus.PENDING.value))
            db.commit()
            logger.info("task %s paused for human review", task_id)
            return

        task.status = TaskStatus.COMPLETED.value
        task.route = result.get("route")
        task.result = result.get("final_answer")
        db.add(task)
        db.commit()
        logger.info("task %s completed (route=%s)", task_id, task.route)

    except Exception as exc:  # noqa: BLE001 - deliberately broad: this is a background job boundary
        db.rollback()
        task = db.get(Task, task_id)
        if task is not None:
            task.status = TaskStatus.FAILED.value
            task.error = str(exc)
            db.add(task)
            db.commit()
        logger.exception("task %s failed", task_id)
        raise
    finally:
        db.close()


@celery_app.task(name="run_agent_graph", bind=False)
def run_agent_graph(task_id: str) -> None:
    _run_or_resume(task_id, resume_value=None)


@celery_app.task(name="resume_agent_graph", bind=False)
def resume_agent_graph(task_id: str, resume_value: str) -> None:
    _run_or_resume(task_id, resume_value=resume_value)
