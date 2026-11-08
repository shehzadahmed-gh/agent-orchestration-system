from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.db.models import Escalation, EscalationStatus


def claim_pending_escalation(db: Session, escalation_id: str, response: str) -> bool:
    """Resolve an escalation if, and only if, it is still pending.

    The status check and the write are one UPDATE ... WHERE status = 'pending',
    so if two responses arrive at the same time only one of them matches the
    row. A separate read-then-write would let both see 'pending' and resume
    the paused graph twice. Returns True if this call resolved it.
    """
    result = db.execute(
        update(Escalation)
        .where(
            Escalation.id == escalation_id,
            Escalation.status == EscalationStatus.PENDING.value,
        )
        .values(
            status=EscalationStatus.RESOLVED.value,
            human_response=response,
            resolved_at=datetime.now(UTC),
        )
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1
