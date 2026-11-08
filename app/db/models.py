"""
Structured state, in PostgreSQL - the counterpart to the semantic memory
layer in app/memory/semantic_memory.py (ChromaDB). Tasks and escalations are
exact, queryable, transactional records (status, timestamps, foreign keys);
semantic memory is fuzzy, similarity-searched recall. Splitting them onto
two different stores means each is used for what it's actually good at,
rather than bolting vector search onto Postgres or trying to do exact
status tracking in Chroma.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    FAILED = "failed"


class EscalationStatus(str, enum.Enum):
    PENDING = "pending"
    RESOLVED = "resolved"


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    input_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default=TaskStatus.PENDING.value, index=True)
    route: Mapped[str | None] = mapped_column(String, nullable=True)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    escalations: Mapped[list["Escalation"]] = relationship(back_populates="task", cascade="all, delete-orphan")


class Escalation(Base):
    __tablename__ = "escalations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default=EscalationStatus.PENDING.value, index=True)
    human_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    task: Mapped["Task"] = relationship(back_populates="escalations")
