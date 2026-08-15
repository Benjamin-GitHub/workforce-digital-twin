from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


COCO_17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
CML_15 = (
    "Head", "ShoulderCenter", "Spine", "LeftShoulder", "LeftElbow", "LeftHand",
    "RightShoulder", "RightElbow", "RightHand", "LeftHip", "LeftKnee", "LeftFoot",
    "RightHip", "RightKnee", "RightFoot",
)
CML_20 = CML_15 + ("HipCenter", "LeftWrist", "LeftAnkle", "RightWrist", "RightAnkle")


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def metadata_value(sample: dict, keys: list[str]):
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def frames_from_tdata(tdata) -> np.ndarray:
    """Accept documented CML frame-major tdata, with clear errors for variants."""
    if isinstance(tdata, dict):
        def order(item):
            digits = "".join(char for char in str(item[0]) if char.isdigit())
            return (0, int(digits)) if digits else (1, str(item[0]))
        tdata = [value for _, value in sorted(tdata.items(), key=order)]
    frames = []
    for frame in tdata or []:
        if isinstance(frame, dict):
            joints = frame.get("joints", frame.get("data", frame))
            if isinstance(joints, dict):
                joints = list(joints.values())
        else:
            joints = frame
        array = np.asarray(joints, dtype=np.float32)
        if array.ndim == 1 and array.size % 3 == 0:
            array = array.reshape(-1, 3)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError("Unrecognized CML tdata frame; expected Vx3 coordinates")
        frames.append(array[:, :3])
    if not frames:
        raise ValueError("CML sample has no usable tdata frames")
    result = np.stack(frames)
    if result.shape[1] not in (15, 20):
        raise ValueError(f"Expected CML 15/20 joints, found {result.shape[1]}")
    return result


def cml_to_coco17(frames: np.ndarray, axes: list[str], confidence: float) -> np.ndarray:
    names = CML_20 if frames.shape[1] == 20 else CML_15
    source = {name: index for index, name in enumerate(names)}
    endpoints = {
        "nose": "Head", "left_shoulder": "LeftShoulder", "right_shoulder": "RightShoulder",
        "left_elbow": "LeftElbow", "right_elbow": "RightElbow", "left_hip": "LeftHip",
        "right_hip": "RightHip", "left_knee": "LeftKnee", "right_knee": "RightKnee",
        "left_wrist": "LeftWrist" if "LeftWrist" in source else "LeftHand",
        "right_wrist": "RightWrist" if "RightWrist" in source else "RightHand",
        "left_ankle": "LeftAnkle" if "LeftAnkle" in source else "LeftFoot",
        "right_ankle": "RightAnkle" if "RightAnkle" in source else "RightFoot",
    }
    axis_index = {"x": 0, "y": 1, "z": 2}
    output = np.zeros((frames.shape[0], 17, 3), dtype=np.float32)
    for target_name, source_name in endpoints.items():
        target = COCO_17.index(target_name)
        output[:, target, 0] = frames[:, source[source_name], axis_index[axes[0]]]
        output[:, target, 1] = frames[:, source[source_name], axis_index[axes[1]]]
        output[:, target, 2] = confidence
    return output


def resample(frames: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    count = max(1, round(len(frames) * target_hz / source_hz))
    old = np.linspace(0.0, 1.0, len(frames))
    new = np.linspace(0.0, 1.0, count)
    result = np.empty((count, *frames.shape[1:]), dtype=np.float32)
    for joint in range(frames.shape[1]):
        for channel in range(frames.shape[2]):
            result[:, joint, channel] = np.interp(new, old, frames[:, joint, channel])
    return result


def normalize(frames: np.ndarray) -> np.ndarray:
    result = frames.copy()
    for frame in result:
        visible = frame[:, 2] > 0
        if not visible.any():
            continue
        hips = visible[11] and visible[12]
        shoulders = visible[5] and visible[6]
        hip_mid = frame[[11, 12], :2].mean(axis=0) if hips else None
        shoulder_mid = frame[[5, 6], :2].mean(axis=0) if shoulders else None
        centre = hip_mid if hip_mid is not None else (shoulder_mid if shoulder_mid is not None else frame[visible, :2].mean(axis=0))
        scale = np.linalg.norm(hip_mid - shoulder_mid) if hips and shoulders else 0.0
        if scale <= 1e-6:
            extent = np.ptp(frame[visible, :2], axis=0)
            scale = np.linalg.norm(extent)
        scale = scale if scale > 1e-6 else 1.0
        frame[visible, :2] = (frame[visible, :2] - centre) / scale
        frame[~visible] = 0.0
    return result


def windows(frames: np.ndarray, size: int, stride: int):
    if len(frames) < size:
        pad = np.repeat(frames[-1:], size - len(frames), axis=0)
        yield np.concatenate((frames, pad), axis=0)
        return
    for start in range(0, len(frames) - size + 1, stride):
        yield frames[start:start + size]


def sample_group(sample: dict, path: Path, cfg: dict) -> str:
    source = metadata_value(sample, cfg["source_keys"]) or "unknown_source"
    session = metadata_value(sample, cfg["session_keys"])
    source_sample = metadata_value(sample, cfg["sample_keys"])
    if session is not None:
        return f"{source}:session:{session}"
    if source_sample is not None:
        return f"{source}:sample:{source_sample}"
    return f"{source}:path:{path.as_posix()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CML JSON to deployment-compatible ST-GCN windows")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/preprocess.yaml")
    parser.add_argument("--classes", type=Path, default=Path(__file__).parent / "configs/classes.yaml")
    parser.add_argument("--inspect-labels", action="store_true")
    args = parser.parse_args()
    cfg, classes_cfg = load_yaml(args.config), load_yaml(args.classes)
    files = sorted(args.input.rglob("*.json"))
    labels = Counter()
    records = []
    for path in files:
        try:
            sample = json.loads(path.read_text(encoding="utf-8-sig"))
            label = metadata_value(sample, cfg["label_keys"])
            labels[str(label)] += 1
            if args.inspect_labels:
                continue
            target = classes_cfg["cml_label_mapping"].get(label)
            if target is None:
                continue
            frames = frames_from_tdata(sample.get("tdata"))
            if len(frames) < cfg["minimum_valid_frames"]:
                continue
            frames = cml_to_coco17(frames, cfg["projection_axes"], cfg["confidence_for_present_joint"])
            frames = normalize(resample(frames, cfg["source_hz"], cfg["target_hz"]))
            for index, window in enumerate(windows(frames, cfg["window_size"], cfg["window_stride"])):
                records.append((window.transpose(2, 0, 1)[..., None], target, sample_group(sample, path, cfg), f"{path}:{index}"))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f"SKIP {path}: {exc}")
    if args.inspect_labels:
        print(json.dumps(dict(labels.most_common()), indent=2))
        return
    if args.output is None:
        parser.error("--output is required unless --inspect-labels is used")
    if not records:
        raise SystemExit("No mapped windows produced. Inspect labels and update configs/classes.yaml.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    class_names = classes_cfg["classes"]
    np.savez_compressed(args.output, x=np.stack([r[0] for r in records]), y=np.array([class_names.index(r[1]) for r in records]), groups=np.array([r[2] for r in records]), sample_ids=np.array([r[3] for r in records]), classes=np.array(class_names))
    print(f"Wrote {len(records)} windows to {args.output}; shape={records[0][0].shape}")


if __name__ == "__main__":
    main()
