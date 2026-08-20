from sqlalchemy import select

from .database import SessionLocal
from .db_models import WorkerEvent
from .models import WorkerState


def save_worker_event(worker: WorkerState) -> WorkerEvent:
    event = WorkerEvent(
        worker_id=worker.worker_id,
        timestamp=worker.timestamp,
        activity=worker.activity.display_activity,
        activity_confidence=max(
            worker.activity.stgcn_confidence,
            worker.activity.baseline_confidence,
        ),
        track_id=worker.tracking.track_id,
        camera_id=worker.tracking.camera_id,
    )

    with SessionLocal() as session:
        session.add(event)
        session.commit()
        session.refresh(event)

    return event


def get_worker_history(
    worker_id: str,
    limit: int = 100,
) -> list[WorkerEvent]:
    with SessionLocal() as session:
        statement = (
            select(WorkerEvent)
            .where(WorkerEvent.worker_id == worker_id)
            .order_by(WorkerEvent.timestamp.desc())
            .limit(limit)
        )

        return list(session.scalars(statement))
