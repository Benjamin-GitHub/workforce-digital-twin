from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from preprocess import normalize, windows


VIDEO_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov"}


def select_person(result) -> np.ndarray | None:
    """Return the largest detected person's COCO-17 x/y/confidence pose."""
    if result.boxes is None or result.keypoints is None or result.keypoints.xy is None:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    points = result.keypoints.xy.detach().cpu().numpy()
    if len(boxes) == 0 or len(points) == 0:
        return None
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    index = int(np.argmax(areas))
    confidence = np.ones(17, dtype=np.float32)
    if result.keypoints.conf is not None:
        confidence = result.keypoints.conf[index].detach().cpu().numpy().astype(np.float32)
    pose = np.column_stack((points[index].astype(np.float32), confidence))
    if pose.shape != (17, 3):
        raise ValueError(f"Expected COCO-17 pose, received {pose.shape}")
    return pose


def extract_video(path: Path, model, target_hz: float, image_size: int, pose_confidence: float,
                  keypoint_confidence: float, device: str,
                  max_seconds: float | None = None) -> tuple[np.ndarray, dict]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open the video")
    source_hz = float(capture.get(cv2.CAP_PROP_FPS))
    if source_hz <= 0:
        capture.release()
        raise ValueError("Video reports an invalid frame rate")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / source_hz if frame_count > 0 else None
    step_seconds = 1.0 / target_hz
    next_sample = 0.0
    poses: list[np.ndarray] = []
    detected = 0
    visible_joints = 0
    visible_body_joints = 0
    body_joint_total = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_index / source_hz
        frame_index += 1
        if max_seconds is not None and timestamp >= max_seconds:
            break
        if timestamp + 1e-9 < next_sample:
            continue
        next_sample += step_seconds
        result = model.predict(frame, imgsz=image_size, conf=pose_confidence, verbose=False, device=device)[0]
        pose = select_person(result)
        if pose is None:
            poses.append(np.zeros((17, 3), dtype=np.float32))
            continue
        pose[pose[:, 2] < keypoint_confidence] = 0.0
        # CML exposes one Head joint but no separate eyes/ears. Mask the four
        # unmatched COCO facial nodes in local data to prevent source-domain
        # leakage while preserving the deployment-compatible V=17 layout.
        pose[1:5] = 0.0
        # CML provides no detector confidence. Use binary presence consistently
        # rather than letting confidence magnitude identify the local source.
        pose[pose[:, 2] > 0.0, 2] = 1.0
        poses.append(pose)
        detected += 1
        visible_joints += int(np.count_nonzero(pose[[0, *range(5, 17)], 2] > 0))
        # Shoulders through ankles: the 12 joints carrying most activity signal.
        visible_body_joints += int(np.count_nonzero(pose[5:17, 2] > 0))
        body_joint_total += 12
    capture.release()
    if not poses:
        raise ValueError("No frames were sampled")
    return np.stack(poses), {
        "source_fps": source_hz,
        "source_frames": frame_count,
        "duration_seconds": duration,
        "sampled_frames": len(poses),
        "detected_frames": detected,
        "detection_rate": detected / len(poses),
        "visible_joint_rate": visible_joints / (detected * 13) if detected else 0.0,
        "visible_body_joint_rate": visible_body_joints / body_joint_total if body_joint_total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create COCO-17 ST-GCN windows from labelled local videos")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Matching YOLO Pose .pt file or exported NCNN model directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-hz", type=float, default=11.0)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--pose-confidence", type=float, default=0.35)
    parser.add_argument("--keypoint-confidence", type=float, default=0.30)
    parser.add_argument("--device", default="0", help="Ultralytics device: 0 for first CUDA GPU, or cpu")
    parser.add_argument("--max-videos", type=int, help="Smoke-test limit")
    parser.add_argument("--max-seconds", type=float, help="Per-video smoke-test limit")
    args = parser.parse_args()

    from ultralytics import YOLO

    classes = ["walking", "standing", "idle", "bending", "carrying", "material_handling"]
    videos = sorted(path for path in args.input.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if args.max_videos is not None:
        videos = videos[:args.max_videos]
    if not videos:
        raise SystemExit(f"No supported videos found below {args.input}")
    model = YOLO(args.model, task="pose")
    records = []
    diagnostics = []
    for path in videos:
        label = path.parent.name.lower()
        if label not in {"standing", "idle"}:
            print(f"SKIP {path}: parent folder must be standing or idle")
            continue
        try:
            frames, diagnostic = extract_video(
                path, model, args.target_hz, args.imgsz, args.pose_confidence,
                args.keypoint_confidence, args.device, args.max_seconds,
            )
            frames = normalize(frames)
            for index, window in enumerate(windows(frames, args.window_size, args.window_stride)):
                records.append((window.transpose(2, 0, 1)[..., None], classes.index(label), f"local:{path.stem}", f"{path}:{index}"))
            diagnostic.update({"file": str(path), "label": label})
            diagnostics.append(diagnostic)
            print(
                f"{path.name}: sampled={diagnostic['sampled_frames']} "
                f"detection_rate={diagnostic['detection_rate']:.1%} "
                f"body_joint_rate={diagnostic['visible_body_joint_rate']:.1%}"
            )
        except (ValueError, RuntimeError) as exc:
            print(f"SKIP {path}: {exc}")
    if not records:
        raise SystemExit("No local-video windows were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=np.stack([record[0] for record in records]),
        y=np.asarray([record[1] for record in records]),
        groups=np.asarray([record[2] for record in records]),
        sample_ids=np.asarray([record[3] for record in records]),
        classes=np.asarray(classes),
    )
    diagnostics_path = args.output.with_suffix(".diagnostics.json")
    import json
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} windows to {args.output}; diagnostics={diagnostics_path}")


if __name__ == "__main__":
    main()
