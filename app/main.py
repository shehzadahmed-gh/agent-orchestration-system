from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db.base import Base, engine
from app.routers import escalations, tasks

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Agent Orchestration System",
    description=(
        "A multi-agent system built with LangGraph, orchestrating tool use, persistent memory, "
        "agent coordination, and human-in-the-loop escalation, with PostgreSQL for structured state, "
        "ChromaDB for semantic memory, and Redis/Celery for background execution."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(tasks.router)
app.include_router(escalations.router)


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "service": settings.app_name,
        "status": "ok",
        "checkpointer_backend": settings.checkpointer_backend,
        "llm_provider": settings.llm_provider,
        "celery_task_always_eager": settings.celery_task_always_eager,
    }


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}
