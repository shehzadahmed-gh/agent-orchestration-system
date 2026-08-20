from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.db.base import SessionLocal
from app.db.models import Escalation, EscalationStatus, Task, TaskStatus
from app.models.schemas import EscalationRespondRequest, EscalationResponse
from app.tasks import resume_agent_graph

router = APIRouter(prefix="/escalations", tags=["escalations"])


@router.get("", response_model=list[EscalationResponse])
def list_escalations(status: str | None = None) -> list[EscalationResponse]:
    with SessionLocal() as db:
        query = db.query(Escalation)
        if status:
            query = query.filter(Escalation.status == status)
        escalations = query.order_by(Escalation.created_at.desc()).all()
        return [EscalationResponse.model_validate(e) for e in escalations]


@router.get("/{escalation_id}", response_model=EscalationResponse)
def get_escalation(escalation_id: str) -> EscalationResponse:
    with SessionLocal() as db:
        escalation = db.get(Escalation, escalation_id)
        if escalation is None:
            raise HTTPException(status_code=404, detail="Escalation not found")
        return EscalationResponse.model_validate(escalation)


@router.post("/{escalation_id}/respond", response_model=EscalationResponse)
def respond_to_escalation(escalation_id: str, request: EscalationRespondRequest) -> EscalationResponse:
    with SessionLocal() as db:
        escalation = db.get(Escalation, escalation_id)
        if escalation is None:
            raise HTTPException(status_code=404, detail="Escalation not found")
        if escalation.status == EscalationStatus.RESOLVED.value:
            raise HTTPException(status_code=409, detail="This escalation has already been resolved")

        escalation.status = EscalationStatus.RESOLVED.value
        escalation.human_response = request.response
        escalation.resolved_at = datetime.now(timezone.utc)
        db.add(escalation)

        task = db.get(Task, escalation.task_id)
        if task is not None:
            task.status = TaskStatus.RUNNING.value
            db.add(task)
        db.commit()
        db.refresh(escalation)
        response = EscalationResponse.model_validate(escalation)
        task_id = escalation.task_id

    # Session closed above before triggering (possibly synchronous, in eager
    # mode) graph resumption - same reasoning as in routers/tasks.py.
    resume_agent_graph.delay(task_id, request.response)

    return response
