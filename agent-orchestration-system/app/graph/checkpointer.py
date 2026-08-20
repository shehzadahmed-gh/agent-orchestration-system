"""
Selects the LangGraph checkpointer backend. Both PostgresSaver and
SqliteSaver are used as context managers, opening a fresh connection per
graph invocation - simple and safe for this service's execution model (one
Celery task = one graph.invoke() call), avoiding the need to manage a
long-lived connection pool across worker processes.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from app.config import Settings


@contextmanager
def get_checkpointer_ctx(settings: Settings):
    if settings.checkpointer_backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        os.makedirs(os.path.dirname(settings.checkpointer_sqlite_path) or ".", exist_ok=True)
        with SqliteSaver.from_conn_string(settings.checkpointer_sqlite_path) as saver:
            yield saver
    else:
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(settings.checkpointer_postgres_conn_string) as saver:
            saver.setup()  # idempotent; creates the checkpoint tables on first run
            yield saver
