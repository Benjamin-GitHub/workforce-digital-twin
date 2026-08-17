"""Contained live inference service for the streaming GRU comparator."""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from .models import PoseState

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]
TRAINING_DIR = REPO_ROOT / "training/gru"
DEFAULT_CHECKPOINT = TRAINING_DIR / "runs/cml_plus_local_sqrt/best.pt"
EXPECTED_CLASSES = [
    "walking", "standing", "idle", "bending", "carrying", "material_handling",
]


@dataclass(frozen=True)
class GRUPrediction:
    worker_id: str
    activity: str
    confidence: float
    probabilities: dict[str, float]
    ready: bool
    observations: int


class GRUInferenceService:
    """Own one streaming predictor and contain load/inference failures."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self.error: str | None = None
        self._latest: dict[str, GRUPrediction] = {}
        self._lock = Lock()

    def load(self) -> None:
        if self.predictor is not None:
            return
        try:
            if str(TRAINING_DIR) not in sys.path:
                sys.path.insert(0, str(TRAINING_DIR))
            from streaming_inference import StreamingGRUPredictor

            predictor = StreamingGRUPredictor(self.checkpoint_path, device="auto")
            if predictor.classes != EXPECTED_CLASSES:
                raise ValueError(
                    f"Unexpected checkpoint class order: {predictor.classes}"
                )
            self.predictor = predictor
            self.error = None
            LOGGER.info(
                "Streaming GRU loaded from %s on %s",
                self.checkpoint_path,
                predictor.device,
            )
        except Exception as exc:
            self.predictor = None
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Streaming GRU could not be loaded")

    def predict(self, worker_id: str, pose: PoseState) -> GRUPrediction | None:
        if self.predictor is None:
            return None
        try:
            values = np.asarray(
                [
                    [
                        point.x,
                        point.y,
                        point.confidence if point.confidence is not None else 0.0,
                    ]
                    for point in pose.keypoints
                ],
                dtype=np.float32,
            )
            with self._lock:
                result = self.predictor.update(worker_id, values)
            prediction = GRUPrediction(
                worker_id=worker_id,
                activity=result["label"],
                confidence=float(result["confidence"]),
                probabilities={
                    name: float(value)
                    for name, value in result["probabilities"].items()
                },
                ready=bool(result["ready"]),
                observations=int(result["observations"]),
            )
            self._latest[worker_id] = prediction
            self.error = None
            return prediction
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Streaming GRU inference failed for worker %s", worker_id)
            return None

    def status(self) -> dict:
        predictor = self.predictor
        return {
            "loaded": predictor is not None,
            "device": str(predictor.device) if predictor is not None else None,
            "checkpoint": str(self.checkpoint_path),
            "classes": list(predictor.classes) if predictor is not None else [],
            "minimum_observations": (
                predictor.minimum_observations if predictor is not None else None
            ),
            "reset_gap_seconds": (
                predictor.reset_gap_seconds if predictor is not None else None
            ),
            "error": self.error,
        }

    def latest(self, worker_id: str) -> dict | None:
        prediction = self._latest.get(worker_id)
        return asdict(prediction) if prediction is not None else None

    def clear_predictions(self) -> None:
        self._latest.clear()
        if self.predictor is not None:
            self.predictor.reset()


gru_service = GRUInferenceService()
