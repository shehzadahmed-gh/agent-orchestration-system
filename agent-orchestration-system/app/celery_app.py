"""
Celery app setup - the "distributed background workloads through Redis and
Celery" piece. `task_always_eager` (driven by CELERY_TASK_ALWAYS_EAGER) lets
the exact same task code run either:

  * synchronously in-process, no broker needed (tests / CI / this sandbox), or
  * genuinely asynchronously via a Redis-backed worker process (docker-compose)

`.delay()` calls in app/routers/tasks.py and app/routers/escalations.py are
unchanged either way - Celery's eager mode is what decides whether that
enqueue actually goes over the wire to Redis or just runs inline.
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
