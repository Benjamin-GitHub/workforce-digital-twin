"""Framework-neutral temporal pose preparation for future ST-GCN inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock

from .models import PoseState


COCO_17_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
DEFAULT_WINDOW_SIZE = 32
CHANNELS = ("x", "y", "confidence")


def _midpoint(points: list[list[float]], left: int, right: int) -> tuple[float, float] | None:
    if points[left][2] <= 0.0 or points[right][2] <= 0.0:
        return None
    return (
        (points[left][0] + points[right][0]) / 2.0,
        (points[left][1] + points[right][1]) / 2.0,
    )


def normalize_pose(pose: PoseState) -> list[list[float]]:
    """Return COCO-17 [x, y, confidence], body-centred and scale-normalized.

    The hip midpoint is the preferred origin. Torso length (shoulder midpoint to
    hip midpoint) is the preferred scale. Visibility-aware fallbacks keep the
    transform usable when either pair is missing. Zero-confidence joints retain
    confidence zero and receive zero coordinates.
    """

    points = [
        [point.x, point.y, point.confidence if point.confidence is not None else 0.0]
        for point in pose.keypoints
    ]
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
        scale = (
            (hip_midpoint[0] - shoulder_midpoint[0]) ** 2
            + (hip_midpoint[1] - shoulder_midpoint[1]) ** 2
        ) ** 0.5
    if scale <= 1e-6:
        width = max(point[0] for point in visible) - min(point[0] for point in visible)
        height = max(point[1] for point in visible) - min(point[1] for point in visible)
        scale = (width * width + height * height) ** 0.5
    if scale <= 1e-6:
        scale = float(max(pose.image_width, pose.image_height))

    normalized = []
    for x, y, confidence in points:
        if confidence <= 0.0:
            normalized.append([0.0, 0.0, 0.0])
        else:
            normalized.append([(x - centre[0]) / scale, (y - centre[1]) / scale, confidence])
    return normalized


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


class TemporalPoseBuffer:
    """Keep an independent, fixed-size normalized sequence for each worker."""

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._frames: dict[str, deque[list[list[float]]]] = {}
        self._last_frame: dict[str, int] = {}
        self._lock = Lock()

    def add(self, worker_id: str, pose: PoseState) -> bool:
        """Add a newer frame; return False for duplicate/out-of-order frames."""
        with self._lock:
            if pose.frame_number <= self._last_frame.get(worker_id, -1):
                return False
            frames = self._frames.setdefault(worker_id, deque(maxlen=self.window_size))
            frames.append(normalize_pose(pose))
            self._last_frame[worker_id] = pose.frame_number
            return True

    def diagnostic(self, worker_id: str) -> SequenceDiagnostic:
        with self._lock:
            count = len(self._frames.get(worker_id, ()))
        ready = count == self.window_size
        return SequenceDiagnostic(
            worker_id=worker_id,
            ready=ready,
            frames_collected=count,
            window_size=self.window_size,
            tensor_shape=[1, 3, self.window_size, 17, 1] if ready else None,
        )

    def tensor(self, worker_id: str) -> list | None:
        with self._lock:
            frames = list(self._frames.get(worker_id, ()))
        if len(frames) != self.window_size:
            return None
        return to_stgcn_tensor(frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._last_frame.clear()


temporal_pose_buffer = TemporalPoseBuffer()
