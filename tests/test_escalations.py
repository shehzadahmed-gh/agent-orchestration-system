import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.escalations import claim_pending_escalation
from app.db.models import Escalation, EscalationStatus, Task


@pytest.fixture()
def session_factory(tmp_path):
    # A file-backed SQLite database so two independent sessions see the same rows.
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


@pytest.fixture()
def escalation_id(session_factory):
    with session_factory() as db:
        task = Task(input_text="Please delete my account.")
        db.add(task)
        db.flush()
        escalation = Escalation(task_id=task.id, question="Approve deletion?")
        db.add(escalation)
        db.commit()
        return escalation.id


def test_first_claim_resolves_the_escalation(session_factory, escalation_id):
    with session_factory() as db:
        assert claim_pending_escalation(db, escalation_id, "approve")
        db.commit()
        escalation = db.get(Escalation, escalation_id)
        assert escalation.status == EscalationStatus.RESOLVED.value
        assert escalation.human_response == "approve"
        assert escalation.resolved_at is not None


def test_second_claim_is_refused(session_factory, escalation_id):
    with session_factory() as db:
        assert claim_pending_escalation(db, escalation_id, "approve")
        db.commit()
    with session_factory() as db:
        assert not claim_pending_escalation(db, escalation_id, "reject")
        assert db.get(Escalation, escalation_id).human_response == "approve"


def test_stale_read_cannot_double_resume(session_factory, escalation_id):
    # Two requests both load the escalation while it is still pending, which
    # is what a check-then-update handler would rely on. Only one claim wins.
    first, second = session_factory(), session_factory()
    try:
        assert first.get(Escalation, escalation_id).status == EscalationStatus.PENDING.value
        assert second.get(Escalation, escalation_id).status == EscalationStatus.PENDING.value

        assert claim_pending_escalation(first, escalation_id, "approve")
        first.commit()
        assert not claim_pending_escalation(second, escalation_id, "reject")
    finally:
        first.close()
        second.close()


def test_unknown_escalation_is_not_claimed(session_factory):
    with session_factory() as db:
        assert not claim_pending_escalation(db, "does-not-exist", "approve")
