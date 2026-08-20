from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TaskCreateRequest(BaseModel):
    input: str


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    input_text: str
    status: str
    route: str | None = None
    result: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class EscalationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    task_id: str
    question: str
    status: str
    human_response: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None


class EscalationRespondRequest(BaseModel):
    response: str
