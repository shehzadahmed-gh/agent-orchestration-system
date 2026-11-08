from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.db.base import SessionLocal
from app.db.models import Task, TaskStatus
from app.models.schemas import TaskCreateRequest, TaskResponse
from app.tasks import run_agent_graph

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskResponse, status_code=202)
def create_task(request: TaskCreateRequest) -> TaskResponse:
    task_id = str(uuid.uuid4())
    with SessionLocal() as db:
        task = Task(id=task_id, input_text=request.input, status=TaskStatus.PENDING.value)
        db.add(task)
        db.commit()
        db.refresh(task)
        response = TaskResponse.model_validate(task)

    # Close the session before .delay(): in eager mode the whole graph runs
    # inline here, and it hung when this request still held a connection.
    run_agent_graph.delay(task_id)

    if get_settings().celery_task_always_eager:
        with SessionLocal() as db:
            task = db.get(Task, task_id)
            response = TaskResponse.model_validate(task)

    return response


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(task_id: str) -> TaskResponse:
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return TaskResponse.model_validate(task)


@router.get("", response_model=list[TaskResponse])
def list_tasks() -> list[TaskResponse]:
    with SessionLocal() as db:
        tasks = db.query(Task).order_by(Task.created_at.desc()).limit(50).all()
        return [TaskResponse.model_validate(t) for t in tasks]
