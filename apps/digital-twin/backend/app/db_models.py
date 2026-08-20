from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
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


class MultimodalSession(Base):
    __tablename__ = "multimodal_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    worker_id: Mapped[str] = mapped_column(String, index=True)
    source_mode: Mapped[str] = mapped_column(String)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_activity: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cadence_hz: Mapped[float] = mapped_column(Float)
    max_samples: Mapped[int] = mapped_column(Integer)
    dropped_vision_samples: Mapped[int] = mapped_column(Integer, default=0)
    dropped_mobile_samples: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_vision_samples: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_mobile_samples: Mapped[int] = mapped_column(Integer, default=0)


class SessionVisionSample(Base):
    __tablename__ = "session_vision_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    worker_id: Mapped[str] = mapped_column(String, index=True)
    vision_timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    backend_receive_time: Mapped[datetime] = mapped_column(DateTime)
    camera_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    camera_id: Mapped[str | None] = mapped_column(String, nullable=True)
    baseline_activity: Mapped[str] = mapped_column(String)
    baseline_confidence: Mapped[float] = mapped_column(Float)
    stgcn_activity: Mapped[str] = mapped_column(String)
    stgcn_confidence: Mapped[float] = mapped_column(Float)
    gru_activity: Mapped[str] = mapped_column(String, default="unknown")
    gru_confidence: Mapped[float] = mapped_column(Float, default=0.0)


class SessionMobileSample(Base):
    __tablename__ = "session_mobile_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    worker_id: Mapped[str] = mapped_column(String, index=True)
    device_id: Mapped[str] = mapped_column(String)
    mobile_timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    backend_receive_time: Mapped[datetime] = mapped_column(DateTime)
    connection_state: Mapped[str] = mapped_column(String)
    mobile_age_ms: Mapped[float] = mapped_column(Float)
    accel_x: Mapped[float] = mapped_column(Float)
    accel_y: Mapped[float] = mapped_column(Float)
    accel_z: Mapped[float] = mapped_column(Float)
    gyro_x: Mapped[float] = mapped_column(Float)
    gyro_y: Mapped[float] = mapped_column(Float)
    gyro_z: Mapped[float] = mapped_column(Float)
    gps_latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_accuracy_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_zone: Mapped[str | None] = mapped_column(String, nullable=True)
    gps_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
