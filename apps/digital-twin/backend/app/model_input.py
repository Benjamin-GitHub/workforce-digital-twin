"""Per-worker pose acceptance, cadence bucketing, and reset diagnostics."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from threading import Lock

from .models import WorkerState


Identity = tuple[str, str | None, int | None]


def _optional_input_hz() -> float | None:
    raw = os.getenv("MODEL_INPUT_HZ", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("MODEL_INPUT_HZ must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError("MODEL_INPUT_HZ must be finite")
    return value if value > 0 else None


def _reset_gap_seconds() -> float:
    raw = os.getenv("MODEL_RESET_GAP_SECONDS", "2.0").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("MODEL_RESET_GAP_SECONDS must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("MODEL_RESET_GAP_SECONDS must be a positive finite number")
    return value


@dataclass
class _WorkerInputState:
    identity: Identity | None = None
    last_source_timestamp: float | None = None
    last_frame_number: int | None = None
    last_cadence_slot: int | None = None
    accepted_count: int = 0
    rejected_count: int = 0
    last_decision_reason: str | None = None


@dataclass(frozen=True)
class ModelInputDecision:
    accepted: bool
    reason: str
    reset_required: bool = False


class ModelInputCoordinator:
    def __init__(
        self,
        input_hz: float | None = None,
        reset_gap_seconds: float | None = None,
        *,
        use_environment: bool = True,
    ) -> None:
        selected_hz = _optional_input_hz() if use_environment else input_hz
        selected_gap = _reset_gap_seconds() if use_environment else reset_gap_seconds
        if selected_hz is not None:
            if not math.isfinite(selected_hz):
                raise ValueError("input_hz must be finite")
            if selected_hz <= 0:
                selected_hz = None
        if selected_gap is None or not math.isfinite(selected_gap) or selected_gap <= 0:
            raise ValueError("reset_gap_seconds must be positive and finite")
        self.input_hz = float(selected_hz) if selected_hz is not None else None
        self.reset_gap_seconds = float(selected_gap)
        self._states: dict[str, _WorkerInputState] = {}
        self._lock = Lock()

    @staticmethod
    def _identity(worker: WorkerState) -> Identity:
        return (worker.source, worker.tracking.camera_id, worker.tracking.track_id)

    def evaluate(self, worker: WorkerState) -> ModelInputDecision:
        if worker.pose is None:
            raise ValueError("A pose-bearing worker update is required")
        source_timestamp = worker.pose.captured_at.timestamp()
        frame_number = worker.pose.frame_number
        identity = self._identity(worker)
        cadence_slot = (
            math.floor(source_timestamp * self.input_hz)
            if self.input_hz is not None
            else None
        )

        with self._lock:
            state = self._states.setdefault(worker.worker_id, _WorkerInputState())
            reason = "accepted"
            reset_required = False

            if state.identity is not None and identity != state.identity:
                reason, reset_required = "identity_change", True
            elif (state.last_source_timestamp is not None
                  and source_timestamp <= state.last_source_timestamp):
                state.rejected_count += 1
                state.last_decision_reason = "duplicate_timestamp"
                return ModelInputDecision(False, "duplicate_timestamp")
            elif (state.last_source_timestamp is not None
                  and source_timestamp - state.last_source_timestamp > self.reset_gap_seconds):
                reason, reset_required = "time_gap", True
            elif (state.last_frame_number is not None
                  and frame_number < state.last_frame_number):
                reason, reset_required = "frame_rewind", True
            elif cadence_slot is not None and cadence_slot == state.last_cadence_slot:
                state.identity = identity
                state.last_source_timestamp = source_timestamp
                state.last_frame_number = frame_number
                state.rejected_count += 1
                state.last_decision_reason = "cadence_slot"
                return ModelInputDecision(False, "cadence_slot")

            state.identity = identity
            state.last_source_timestamp = source_timestamp
            state.last_frame_number = frame_number
            state.last_cadence_slot = cadence_slot
            state.accepted_count += 1
            state.last_decision_reason = reason
            return ModelInputDecision(True, reason, reset_required)

    def reset(self, worker_id: str, reason: str = "session_start") -> None:
        with self._lock:
            self._states[worker_id] = _WorkerInputState(last_decision_reason=reason)

    def diagnostic(self, worker_id: str) -> dict:
        with self._lock:
            state = self._states.get(worker_id, _WorkerInputState())
            identity = state.identity
            return {
                "worker_id": worker_id,
                "configured_input_hz": self.input_hz,
                "reset_gap_seconds": self.reset_gap_seconds,
                "accepted_count": state.accepted_count,
                "rejected_count": state.rejected_count,
                "last_decision_reason": state.last_decision_reason,
                "current_identity": (
                    {
                        "source": identity[0],
                        "camera_id": identity[1],
                        "track_id": identity[2],
                    }
                    if identity is not None else None
                ),
                "last_source_timestamp": state.last_source_timestamp,
                "last_frame_number": state.last_frame_number,
                "last_cadence_slot": state.last_cadence_slot,
            }


model_input_coordinator = ModelInputCoordinator()
