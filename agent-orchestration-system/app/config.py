"""
Central configuration for the agent orchestration service.

As in the RAG project, every backend is chosen via environment variables so
the same code runs fully offline (SQLite checkpointer, in-process Celery
execution, a deterministic local "LLM") for local dev/tests, or against real
infrastructure (Postgres, Redis + a real Celery worker, GPT-4o/Claude
Sonnet) via docker-compose in production. See .env.example.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "agent-orchestration-system"
    environment: str = "development"

    # --- structured state (PostgreSQL) ---
    database_url: str = "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/agentdb?connect_timeout=10"

    # --- LangGraph checkpointer: "postgres" (production, docker-compose) or "sqlite" (local dev/tests) ---
    checkpointer_backend: Literal["postgres", "sqlite"] = "postgres"
    checkpointer_postgres_conn_string: str = "postgresql://postgres:postgres@127.0.0.1:5432/agentdb?connect_timeout=10"
    checkpointer_sqlite_path: str = "./data/checkpoints.sqlite"

    # --- semantic memory (ChromaDB) ---
    chroma_url: str | None = None
    chroma_local_path: str | None = "./data/chroma_local"
    semantic_memory_collection: str = "agent_semantic_memory"
    semantic_memory_top_k: int = 3

    # --- background execution (Redis + Celery) ---
    redis_url: str = "redis://localhost:6379/0"
    # When true, tasks run synchronously in-process instead of via a Celery
    # worker + Redis broker - used by tests/CI so the graph logic is
    # exercisable without standing up infrastructure. docker-compose sets
    # this to false so real async execution is used in "production".
    celery_task_always_eager: bool = False

    # --- embeddings for semantic memory (same offline-by-default pattern as the RAG project) ---
    embedding_backend: Literal["local", "openai"] = "local"
    embedding_dimension: int = 384
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: str | None = None

    # --- LLM backend for the graph's reasoning nodes ---
    # "local"     -> deterministic rule-based provider, no API key needed
    # "openai"    -> GPT-4o
    # "anthropic" -> Claude Sonnet
    llm_provider: Literal["local", "openai", "anthropic"] = "local"
    openai_chat_model: str = "gpt-4o"
    anthropic_chat_model: str = "claude-sonnet-5"
    anthropic_api_key: str | None = None

    # --- tool integration with the rag-hybrid-search project (optional) ---
    rag_service_url: str | None = None

    # --- human-in-the-loop ---
    escalation_keywords: str = "delete,wire transfer,payment,refund,cancel subscription,irreversible"


@lru_cache
def get_settings() -> Settings:
    return Settings()
