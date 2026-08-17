from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPECTED_CLASSES = (
    "walking",
    "standing",
    "idle",
    "bending",
    "carrying",
    "material_handling",
)


def _as_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


@dataclass(frozen=True)
class PoseArchive:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sample_ids: np.ndarray
    classes: list[str]


def load_pose_archive(path: Path, temporal_stride: int = 2) -> PoseArchive:
    if temporal_stride < 1:
        raise ValueError("temporal_stride must be at least 1")
    with np.load(path) as archive:
        required = {"x", "y", "groups", "classes"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Archive is missing fields: {sorted(missing)}")
        raw_x = archive["x"]
        y = archive["y"].astype(np.int64, copy=False)
        groups = np.asarray([_as_text(item) for item in archive["groups"]])
        classes = [_as_text(item) for item in archive["classes"].tolist()]
        sample_ids = (
            np.asarray([_as_text(item) for item in archive["sample_ids"]])
            if "sample_ids" in archive.files
            else np.asarray([f"sample:{i}" for i in range(len(y))])
        )

    if raw_x.ndim != 5 or raw_x.shape[1] != 3 or raw_x.shape[3:] != (17, 1):
        raise ValueError(
            "Expected ST-GCN archive shape (N,3,T,17,1); "
            f"received {raw_x.shape}"
        )
    if classes != list(EXPECTED_CLASSES):
        raise ValueError(
            "Class order must remain " + repr(list(EXPECTED_CLASSES)) + f"; found {classes}"
        )
    if not (len(raw_x) == len(y) == len(groups) == len(sample_ids)):
        raise ValueError("x, y, groups, and sample_ids lengths differ")
    if not np.isfinite(raw_x).all():
        raise ValueError("Pose archive contains NaN or infinite values")

    # N,C,T,V,M -> N,T,C,V -> N,T,51. The stride-2 default converts the
    # existing ~11 Hz windows to ~5.5 Hz, close to the 5.43 Hz live cadence.
    x = raw_x[:, :, ::temporal_stride, :, 0]
    x = x.transpose(0, 2, 1, 3).reshape(len(raw_x), x.shape[2], 51)
    return PoseArchive(
        x=np.ascontiguousarray(x, dtype=np.float32),
        y=y,
        groups=groups,
        sample_ids=sample_ids,
        classes=classes,
    )


def group_safe_split(
    y: np.ndarray,
    groups: np.ndarray,
    fractions: list[float],
    seed: int,
    attempts: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Required only for training-time dataset splitting.
    from sklearn.model_selection import GroupShuffleSplit

    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must contain train/val/test values summing to 1")
    labels = set(np.unique(y).tolist())
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three independent groups are required")

    for attempt in range(attempts):
        current_seed = seed + attempt
        first = GroupShuffleSplit(1, train_size=fractions[0], random_state=current_seed)
        train, remainder = next(first.split(y, y, groups))
        relative_val = fractions[1] / (fractions[1] + fractions[2])
        second = GroupShuffleSplit(1, train_size=relative_val, random_state=current_seed + 10_000)
        val_rel, test_rel = next(
            second.split(y[remainder], y[remainder], groups[remainder])
        )
        val, test = remainder[val_rel], remainder[test_rel]
        if all(set(np.unique(y[index]).tolist()) == labels for index in (train, val, test)):
            return train, val, test
    raise ValueError(
        "Could not create grouped train/val/test splits containing every class. "
        "Add independent groups for rare classes or supply a pre-defined split."
    )


def normalize_live_coco17(frame: np.ndarray) -> np.ndarray:
    """Apply the CML-aligned live preprocessing and return 51 channel-major features."""
    pose = np.asarray(frame, dtype=np.float32).copy()
    if pose.shape != (17, 3):
        raise ValueError(f"Expected a (17,3) x/y/confidence pose; received {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError("Live pose contains NaN or infinite values")

    pose[1:5] = 0.0  # CML has no separate eye/ear nodes.
    visible = pose[:, 2] > 0
    pose[:, 2] = visible.astype(np.float32)  # Match CML binary presence.
    if not visible.any():
        return np.zeros(51, dtype=np.float32)

    hips = bool(visible[11] and visible[12])
    shoulders = bool(visible[5] and visible[6])
    hip_mid = pose[[11, 12], :2].mean(axis=0) if hips else None
    shoulder_mid = pose[[5, 6], :2].mean(axis=0) if shoulders else None
    if hip_mid is not None:
        centre = hip_mid
    elif shoulder_mid is not None:
        centre = shoulder_mid
    else:
        centre = pose[visible, :2].mean(axis=0)

    scale = (
        float(np.linalg.norm(hip_mid - shoulder_mid))
        if hip_mid is not None and shoulder_mid is not None
        else 0.0
    )
    if scale <= 1e-6:
        scale = float(np.linalg.norm(np.ptp(pose[visible, :2], axis=0)))
    if scale <= 1e-6:
        scale = 1.0
    pose[visible, :2] = (pose[visible, :2] - centre) / scale
    pose[~visible] = 0.0
    return np.ascontiguousarray(pose.T.reshape(51), dtype=np.float32)
