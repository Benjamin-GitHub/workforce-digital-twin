"""Bounded, unfused vision/mobile recording and timestamp diagnostics."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from .database import DATA_DIR, SessionLocal, ensure_session_vision_gru_columns
from .db_models import MultimodalSession, SessionMobileSample, SessionVisionSample
from .models import MobileTelemetry, WorkerState


EXPORT_DIR = DATA_DIR / "session_exports"

# Preserve existing evidence while extending the live recorder schema.
ensure_session_vision_gru_columns()


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc)


def _naive_utc(value: datetime) -> datetime:
    return _utc(value).replace(tzinfo=None)


def _iso(value: datetime | None) -> str | None:
    return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z") if value else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


class SessionRecorder:
    def __init__(self) -> None:
        self.active_session_id: str | None = None
        self._last_vision_received: datetime | None = None
        self._last_mobile_received: datetime | None = None
        self._last_vision_source: datetime | None = None
        self._last_mobile_source: datetime | None = None

    def start(self, worker_id: str, source_mode: str, notes: str | None,
              expected_activity: str | None, cadence_hz: float, max_samples: int) -> dict:
        if self.active_session_id is not None:
            raise ValueError(f"session '{self.active_session_id}' is already active")
        session_id = f"session-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        row = MultimodalSession(
            session_id=session_id, worker_id=worker_id, source_mode=source_mode,
            notes=notes, expected_activity=expected_activity,
            started_at=_naive_utc(datetime.now(timezone.utc)), cadence_hz=cadence_hz,
            max_samples=max_samples,
        )
        with SessionLocal() as db:
            db.add(row)
            db.commit()
        self.active_session_id = session_id
        self._last_vision_received = self._last_mobile_received = None
        self._last_vision_source = self._last_mobile_source = None
        return self.status()

    def stop(self) -> dict:
        if self.active_session_id is None:
            raise ValueError("no session is active")
        session_id = self.active_session_id
        with SessionLocal() as db:
            row = db.get(MultimodalSession, session_id)
            row.ended_at = _naive_utc(datetime.now(timezone.utc))
            db.commit()
        self.active_session_id = None
        return self.summary(session_id)

    def status(self) -> dict:
        if self.active_session_id is None:
            return {"active": False, "session_id": None}
        result = self.summary(self.active_session_id)
        result["active"] = True
        return result

    def _accept(self, db, session, received: datetime, source: datetime,
                kind: str, count: int) -> bool:
        last_received = self._last_vision_received if kind == "vision" else self._last_mobile_received
        last_source = self._last_vision_source if kind == "vision" else self._last_mobile_source
        duplicate_field = f"duplicate_{kind}_samples"
        dropped_field = f"dropped_{kind}_samples"
        if last_source is not None and _utc(source) == _utc(last_source):
            setattr(session, duplicate_field, getattr(session, duplicate_field) + 1)
            db.commit()
            return False
        if count >= session.max_samples:
            setattr(session, dropped_field, getattr(session, dropped_field) + 1)
            db.commit()
            return False
        if last_received is not None and (_utc(received) - _utc(last_received)).total_seconds() < 1 / session.cadence_hz:
            setattr(session, dropped_field, getattr(session, dropped_field) + 1)
            db.commit()
            return False
        return True

    def record_vision(self, worker: WorkerState, received: datetime | None = None) -> bool:
        if self.active_session_id is None:
            return False
        received = received or datetime.now(timezone.utc)
        with SessionLocal() as db:
            session = db.get(MultimodalSession, self.active_session_id)
            if session is None or session.worker_id != worker.worker_id:
                return False
            count = db.query(SessionVisionSample).filter_by(session_id=session.session_id).count()
            if not self._accept(db, session, received, worker.timestamp, "vision", count):
                return False
            db.add(SessionVisionSample(
                session_id=session.session_id, worker_id=worker.worker_id,
                vision_timestamp=_naive_utc(worker.timestamp), backend_receive_time=_naive_utc(received),
                camera_track_id=worker.tracking.track_id, camera_id=worker.tracking.camera_id,
                baseline_activity=worker.activity.baseline,
                baseline_confidence=worker.activity.baseline_confidence,
                stgcn_activity=worker.activity.stgcn,
                stgcn_confidence=worker.activity.stgcn_confidence,
                gru_activity=worker.activity.gru,
                gru_confidence=worker.activity.gru_confidence,
            ))
            db.commit()
        self._last_vision_received, self._last_vision_source = received, worker.timestamp
        return True

    def record_mobile(self, mobile: MobileTelemetry, received: datetime | None = None) -> bool:
        if self.active_session_id is None:
            return False
        received = received or mobile.received_at
        with SessionLocal() as db:
            session = db.get(MultimodalSession, self.active_session_id)
            if session is None or session.worker_id != mobile.worker_id:
                return False
            count = db.query(SessionMobileSample).filter_by(session_id=session.session_id).count()
            if not self._accept(db, session, received, mobile.timestamp, "mobile", count):
                return False
            # Keep the sign: a negative value is useful evidence that the phone
            # clock is ahead of the backend rather than a value to conceal.
            age_ms = (_utc(received) - _utc(mobile.timestamp)).total_seconds() * 1000
            location = mobile.location
            db.add(SessionMobileSample(
                session_id=session.session_id, worker_id=mobile.worker_id, device_id=mobile.device_id,
                mobile_timestamp=_naive_utc(mobile.timestamp), backend_receive_time=_naive_utc(received),
                connection_state=mobile.connection_state, mobile_age_ms=age_ms,
                accel_x=mobile.accelerometer.x, accel_y=mobile.accelerometer.y, accel_z=mobile.accelerometer.z,
                gyro_x=mobile.gyroscope.x, gyro_y=mobile.gyroscope.y, gyro_z=mobile.gyroscope.z,
                gps_latitude=location.latitude, gps_longitude=location.longitude,
                gps_accuracy_m=location.accuracy_m, gps_zone=location.zone, gps_enabled=location.gps_enabled,
            ))
            db.commit()
        self._last_mobile_received, self._last_mobile_source = received, mobile.timestamp
        return True

    def _rows(self, session_id: str) -> tuple[MultimodalSession, list[dict]]:
        with SessionLocal() as db:
            session = db.get(MultimodalSession, session_id)
            if session is None:
                raise KeyError(session_id)
            visions = list(db.scalars(select(SessionVisionSample).where(
                SessionVisionSample.session_id == session_id).order_by(SessionVisionSample.vision_timestamp)))
            mobiles = list(db.scalars(select(SessionMobileSample).where(
                SessionMobileSample.session_id == session_id).order_by(SessionMobileSample.mobile_timestamp)))
            db.expunge(session)
            for item in visions + mobiles:
                db.expunge(item)
        tolerance_ms = float(os.getenv("SESSION_ALIGNMENT_TOLERANCE_MS", "1000"))
        rows = []
        for vision in visions:
            nearest = min(mobiles, key=lambda m: abs((vision.vision_timestamp - m.mobile_timestamp).total_seconds()), default=None)
            delta_ms = (vision.vision_timestamp - nearest.mobile_timestamp).total_seconds() * 1000 if nearest else None
            mobile = nearest if nearest is not None and abs(delta_ms) <= tolerance_ms else None
            vision_age = (vision.backend_receive_time - vision.vision_timestamp).total_seconds() * 1000
            rows.append({
                "session_id": session_id, "worker_id": vision.worker_id,
                "device_id": mobile.device_id if mobile else None,
                "camera_track_id": vision.camera_track_id, "camera_id": vision.camera_id,
                "baseline_activity": vision.baseline_activity, "baseline_confidence": vision.baseline_confidence,
                "stgcn_activity": vision.stgcn_activity, "stgcn_confidence": vision.stgcn_confidence,
                "gru_activity": vision.gru_activity, "gru_confidence": vision.gru_confidence,
                "vision_timestamp": _iso(vision.vision_timestamp),
                "android_timestamp": _iso(mobile.mobile_timestamp) if mobile else None,
                "backend_receive_time": _iso(vision.backend_receive_time),
                "mobile_backend_receive_time": _iso(mobile.backend_receive_time) if mobile else None,
                "vision_age_ms": round(vision_age, 3),
                "mobile_age_ms": round(mobile.mobile_age_ms, 3) if mobile else None,
                "source_time_delta_ms": round(delta_ms, 3) if mobile else None,
                "mobile_missing": mobile is None,
                "mobile_stale": mobile is None or mobile.connection_state != "connected",
                "connection_state": mobile.connection_state if mobile else "missing",
                "accelerometer_x": mobile.accel_x if mobile else None,
                "accelerometer_y": mobile.accel_y if mobile else None,
                "accelerometer_z": mobile.accel_z if mobile else None,
                "gyroscope_x": mobile.gyro_x if mobile else None,
                "gyroscope_y": mobile.gyro_y if mobile else None,
                "gyroscope_z": mobile.gyro_z if mobile else None,
                "gps_latitude": mobile.gps_latitude if mobile else None,
                "gps_longitude": mobile.gps_longitude if mobile else None,
                "gps_accuracy_m": mobile.gps_accuracy_m if mobile else None,
                "gps_zone": mobile.gps_zone if mobile else None,
                "gps_missing": mobile is None or mobile.gps_latitude is None or mobile.gps_longitude is None,
            })
        return session, rows

    def summary(self, session_id: str) -> dict:
        session, rows = self._rows(session_id)
        deltas = [abs(row["source_time_delta_ms"]) for row in rows if row["source_time_delta_ms"] is not None]
        with SessionLocal() as db:
            mobile_count = db.query(SessionMobileSample).filter_by(session_id=session_id).count()
        return {
            "session_id": session.session_id, "worker_id": session.worker_id,
            "source_mode": session.source_mode, "notes": session.notes,
            "expected_activity": session.expected_activity,
            "start_time": _iso(session.started_at), "end_time": _iso(session.ended_at),
            "cadence_hz": session.cadence_hz, "max_samples_per_source": session.max_samples,
            "sample_count": len(rows), "vision_sample_count": len(rows), "mobile_sample_count": mobile_count,
            "median_abs_vision_mobile_delta_ms": round(statistics.median(deltas), 3) if deltas else None,
            "p95_abs_vision_mobile_delta_ms": _percentile(deltas, .95),
            "stale_count": sum(row["mobile_stale"] for row in rows),
            "missing_mobile_count": sum(row["mobile_missing"] for row in rows),
            "missing_gps_count": sum(row["gps_missing"] for row in rows),
            "dropped_sample_count": session.dropped_vision_samples + session.dropped_mobile_samples,
            "dropped_vision_samples": session.dropped_vision_samples,
            "dropped_mobile_samples": session.dropped_mobile_samples,
            "duplicate_sample_count": session.duplicate_vision_samples + session.duplicate_mobile_samples,
            "duplicate_vision_samples": session.duplicate_vision_samples,
            "duplicate_mobile_samples": session.duplicate_mobile_samples,
        }

    def export(self, session_id: str, export_format: str) -> Path:
        session, rows = self._rows(session_id)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = EXPORT_DIR / f"{session_id}.{export_format}"
        if export_format == "json":
            path.write_text(json.dumps({"metadata": self.summary(session_id), "samples": rows}, indent=2), encoding="utf-8")
        elif export_format == "csv":
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["session_id", "worker_id"])
                writer.writeheader()
                writer.writerows(rows)
        else:
            raise ValueError("format must be csv or json")
        return path


session_recorder = SessionRecorder()
