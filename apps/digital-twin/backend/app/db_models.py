from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class WorkerEvent(Base):
    __tablename__ = "worker_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    worker_id: Mapped[str] = mapped_column(
        String,
        index=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        index=True,
    )

    activity: Mapped[str] = mapped_column(
        String,
    )

    activity_confidence: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    track_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    camera_id: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )
