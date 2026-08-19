"""Temporal pose preparation and live ST-GCN inference."""

from __future__ import annotations

import logging
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

from .models import PoseState

LOGGER = logging.getLogger(__name__)
COCO_17_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
DEFAULT_WINDOW_SIZE = 32
CHANNELS = ("x", "y", "confidence")
LIVE_KEYPOINT_CONFIDENCE = 0.30
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = REPO_ROOT / "training/stgcn/runs/cml_plus_local_sqrt/best.pt"
EXPECTED_CLASSES = [
    "walking", "standing", "idle", "bending", "carrying", "material_handling",
]


def _checkpoint_path(path: Path | str | None) -> Path:
    selected = path if path is not None else os.getenv("STGCN_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    resolved = Path(selected).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _midpoint(points: list[list[float]], left: int, right: int) -> tuple[float, float] | None:
    if points[left][2] <= 0.0 or points[right][2] <= 0.0:
        return None
    return ((points[left][0] + points[right][0]) / 2.0,
            (points[left][1] + points[right][1]) / 2.0)


def normalize_pose(pose: PoseState) -> list[list[float]]:
    """Return COCO-17 [x, y, confidence], body-centred and scale-normalized."""
    points = [[point.x, point.y, point.confidence if point.confidence is not None else 0.0]
              for point in pose.keypoints]
    for point in points:
        if point[2] < LIVE_KEYPOINT_CONFIDENCE:
            point[:] = [0.0, 0.0, 0.0]
    for index in range(1, 5):
        points[index] = [0.0, 0.0, 0.0]
    for point in points:
        point[2] = 1.0 if point[2] > 0.0 else 0.0
    visible = [point for point in points if point[2] > 0.0]
    if not visible:
        return [[0.0, 0.0, 0.0] for _ in points]
    hip_midpoint = _midpoint(points, 11, 12)
    shoulder_midpoint = _midpoint(points, 5, 6)
    centre = hip_midpoint or shoulder_midpoint or (
        sum(point[0] for point in visible) / len(visible),
        sum(point[1] for point in visible) / len(visible),
    )
    scale = 0.0
    if hip_midpoint is not None and shoulder_midpoint is not None:
        scale = ((hip_midpoint[0] - shoulder_midpoint[0]) ** 2
                 + (hip_midpoint[1] - shoulder_midpoint[1]) ** 2) ** 0.5
    if scale <= 1e-6:
        width = max(point[0] for point in visible) - min(point[0] for point in visible)
        height = max(point[1] for point in visible) - min(point[1] for point in visible)
        scale = (width * width + height * height) ** 0.5
    if scale <= 1e-6:
        scale = 1.0
    return [[(x - centre[0]) / scale, (y - centre[1]) / scale, confidence]
            if confidence > 0.0 else [0.0, 0.0, 0.0]
            for x, y, confidence in points]


def to_stgcn_tensor(frames: list[list[list[float]]]) -> list:
    """Map T,V,C frames to the ST-GCN convention N,C,T,V,M."""
    return [[[[[frames[t][v][c]] for v in range(17)] for t in range(len(frames))]
             for c in range(3)]]


@dataclass(frozen=True)
class SequenceDiagnostic:
    worker_id: str
    ready: bool
    frames_collected: int
    window_size: int
    tensor_shape: list[int] | None
    layout: str = "coco_17"
    channels: tuple[str, ...] = CHANNELS


@dataclass(frozen=True)
class STGCNPrediction:
    worker_id: str
    activity: str
    confidence: float
    probabilities: dict[str, float]


class TemporalPoseBuffer:
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._frames: dict[str, deque[list[list[float]]]] = {}
        self._last_frame: dict[str, int] = {}
        self._lock = Lock()

    def configure(self, window_size: int) -> None:
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be positive")
        with self._lock:
            self.window_size = window_size
            self._frames.clear()
            self._last_frame.clear()

    def add(self, worker_id: str, pose: PoseState) -> bool:
        with self._lock:
            if pose.frame_number <= self._last_frame.get(worker_id, -1):
                return False
            self._frames.setdefault(worker_id, deque(maxlen=self.window_size)).append(normalize_pose(pose))
            self._last_frame[worker_id] = pose.frame_number
            return True

    def diagnostic(self, worker_id: str) -> SequenceDiagnostic:
        with self._lock:
            count = len(self._frames.get(worker_id, ()))
        ready = count == self.window_size
        return SequenceDiagnostic(worker_id, ready, count, self.window_size,
                                  [1, 3, self.window_size, 17, 1] if ready else None)

    def tensor(self, worker_id: str) -> list | None:
        with self._lock:
            frames = list(self._frames.get(worker_id, ()))
        return to_stgcn_tensor(frames) if len(frames) == self.window_size else None

    def clear(self, worker_id: str | None = None) -> None:
        with self._lock:
            if worker_id is None:
                self._frames.clear()
                self._last_frame.clear()
            else:
                self._frames.pop(worker_id, None)
                self._last_frame.pop(worker_id, None)


class STGCNInferenceService:
    """Own the single model instance and contain all load/inference failures."""

    def __init__(self, checkpoint_path: Path | str | None = None):
        self.checkpoint_path = _checkpoint_path(checkpoint_path)
        self.model = None
        self.device = None
        self.classes: list[str] = []
        self.window_size = DEFAULT_WINDOW_SIZE
        self.source_pose_hz: float | None = None
        self.error: str | None = None
        self._latest: dict[str, STGCNPrediction] = {}
        self._lock = Lock()

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from training.stgcn.stgcn_model import STGCN

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            classes, input_shape, config = (list(checkpoint["classes"]),
                                            list(checkpoint["input_shape"]),
                                            checkpoint["config"])
            if classes != EXPECTED_CLASSES:
                raise ValueError(f"Unexpected checkpoint class order: {classes}")
            if (len(input_shape) != 4 or input_shape[0] != 3
                    or not isinstance(input_shape[1], int) or input_shape[1] <= 0
                    or input_shape[2:] != [17, 1]):
                raise ValueError(f"Unexpected checkpoint input shape: {input_shape}")
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            model = STGCN(len(classes), tuple(config["hidden_channels"]), float(config["dropout"]))
            model.load_state_dict(checkpoint["model_state"])
            model.eval().to(device)
            self.model, self.device, self.classes = model, device, classes
            self.window_size, self.error = input_shape[1], None
            source_pose_hz = config.get("source_pose_hz")
            self.source_pose_hz = float(source_pose_hz) if source_pose_hz is not None else None
            LOGGER.info("ST-GCN loaded from %s on %s", self.checkpoint_path, device)
        except Exception as exc:
            self.model = None
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("ST-GCN could not be loaded")

    def predict(self, worker_id: str, values: list) -> STGCNPrediction | None:
        if self.model is None or self.device is None:
            return None
        try:
            import torch
            tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
            if tuple(tensor.shape) != (1, 3, self.window_size, 17, 1):
                raise ValueError(f"Unexpected live input shape: {tuple(tensor.shape)}")
            tensor[:, :, :, 1:5, :] = 0.0
            tensor[:, 2, :, :, :] = (tensor[:, 2, :, :, :] > 0.0).float()
            with self._lock, torch.inference_mode():
                probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu()
            index = int(probabilities.argmax().item())
            prediction = STGCNPrediction(
                worker_id, self.classes[index], float(probabilities[index].item()),
                {name: float(probabilities[i].item()) for i, name in enumerate(self.classes)},
            )
            self._latest[worker_id], self.error = prediction, None
            return prediction
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("ST-GCN inference failed for worker %s", worker_id)
            return None

    def status(self) -> dict:
        return {"loaded": self.model is not None,
                "device": str(self.device) if self.device is not None else None,
                "checkpoint": str(self.checkpoint_path), "classes": self.classes,
                "window_size": self.window_size, "source_pose_hz": self.source_pose_hz,
                "error": self.error}

    def latest(self, worker_id: str) -> dict | None:
        prediction = self._latest.get(worker_id)
        return asdict(prediction) if prediction is not None else None

    def clear_predictions(self, worker_id: str | None = None) -> None:
        if worker_id is None:
            self._latest.clear()
        else:
            self._latest.pop(worker_id, None)


temporal_pose_buffer = TemporalPoseBuffer()
stgcn_service = STGCNInferenceService()
