"""Celery app. With CELERY_TASK_ALWAYS_EAGER=true, .delay() runs tasks inline
with no broker (local runs and tests). Otherwise tasks go through Redis to a
separate worker process.
"""

from __future__ import annotations

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "agent_orchestration",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=True,
    worker_hijack_root_logger=False,
)
