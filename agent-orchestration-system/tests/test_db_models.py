import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import Escalation, EscalationStatus, Task, TaskStatus


@pytest.fixture()
def session():
    # A throwaway in-memory database, independent of the Postgres-backed
    # engine in app/db/base.py - this is a unit test for the model
    # definitions/relationships, not an integration test of the real
    # connection.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_task_defaults(session):
    task = Task(input_text="What is 2 + 2?")
    session.add(task)
    session.commit()
    session.refresh(task)

    assert task.id  # uuid default populated
    assert task.status == TaskStatus.PENDING.value
    assert task.route is None
    assert task.result is None


def test_escalation_linked_to_task(session):
    task = Task(input_text="Please delete my account.")
    session.add(task)
    session.commit()

    escalation = Escalation(task_id=task.id, question="Approve deletion?")
    session.add(escalation)
    session.commit()
    session.refresh(task)

    assert len(task.escalations) == 1
    assert task.escalations[0].status == EscalationStatus.PENDING.value
    assert task.escalations[0].question == "Approve deletion?"


def test_task_status_transitions(session):
    task = Task(input_text="test")
    session.add(task)
    session.commit()

    task.status = TaskStatus.RUNNING.value
    session.commit()
    assert task.status == "running"

    task.status = TaskStatus.COMPLETED.value
    task.result = "the answer"
    session.commit()
    session.refresh(task)
    assert task.status == "completed"
    assert task.result == "the answer"
