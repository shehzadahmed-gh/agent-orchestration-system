"""Service configuration, read from environment variables or a .env file.

The defaults run everything locally (rule-based provider, local Chroma,
in-process Celery if CELERY_TASK_ALWAYS_EAGER=true). docker-compose.yml sets
the values for the full stack. See .env.example for every option.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.llm_provider import DEFAULT_ESCALATION_KEYWORDS


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
    # true: .delay() runs the task inline, so no broker or worker is needed
    # (local runs and tests). docker-compose sets false and runs a worker.
    celery_task_always_eager: bool = False

    # --- embeddings for semantic memory ---
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

    # --- knowledge_base_search tool: base URL of a running rag-hybrid-search service ---
    rag_service_url: str | None = None

    # --- human-in-the-loop (comma-separated, used by the local rule-based provider) ---
    escalation_keywords: str = DEFAULT_ESCALATION_KEYWORDS


@lru_cache
def get_settings() -> Settings:
    return Settings()
