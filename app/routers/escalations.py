from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.base import SessionLocal
from app.db.escalations import claim_pending_escalation
from app.db.models import Escalation, Task, TaskStatus
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
        if db.get(Escalation, escalation_id) is None:
            raise HTTPException(status_code=404, detail="Escalation not found")
        if not claim_pending_escalation(db, escalation_id, request.response):
            db.rollback()
            raise HTTPException(status_code=409, detail="This escalation has already been resolved")

        escalation = db.get(Escalation, escalation_id, populate_existing=True)
        task = db.get(Task, escalation.task_id)
        if task is not None:
            task.status = TaskStatus.RUNNING.value
        db.commit()
        db.refresh(escalation)
        response = EscalationResponse.model_validate(escalation)
        task_id = escalation.task_id

    # The session is closed before resuming. In eager mode .delay() runs the
    # whole graph inline, and it must not do that while this request still
    # holds a connection and an open transaction (see "Debugging notes" in the README).
    resume_agent_graph.delay(task_id, request.response)
    return response
