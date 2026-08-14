#!/usr/bin/env python3

import argparse
import csv
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from src.activity.temporal_buffer import TemporalPoseBuffer
from src.activity.feature_extractor import extract_pose_features
from src.activity.activity_classifier import ActivityClassifier
from src.activity.state_smoother import StateSmoother
from src.activity.material_handling_detector import (
    MaterialHandlingDetector,
)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_activity(
    frame,
    box,
    track_id,
    activity,
    confidence,
):
    x1, y1, x2, y2 = map(int, box)

    label = (
        f"ID {track_id} | "
        f"{activity} | "
        f"{confidence:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():

    # --------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO pose NCNN model",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Video file, stream URL, or camera",
    )

    parser.add_argument(
        "--config",
        default="config/activity.yaml",
        help="Activity configuration YAML file",
    )

    parser.add_argument(
        "--output",
        default="results/activity/output.mp4",
        help="Output annotated video",
    )

    parser.add_argument(
        "--csv",
        default="results/activity/activity_log.csv",
        help="Output activity CSV log",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live OpenCV window",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    config = load_config(args.config)

    pose_config = config.get("pose", {})
    temporal_config = config.get("temporal", {})
    activity_config = config.get("activity", {})
    smoothing_config = config.get("smoothing", {})
    material_config = config.get(
        "material_handling",
        {},
    )

    imgsz = pose_config.get("imgsz", 320)
    pose_conf = pose_config.get("confidence", 0.35)
    
    keypoint_conf = pose_config.get(
        "keypoint_confidence",
        0.30,
    )

    buffer_size = temporal_config.get(
        "buffer_size",
        20,
    )
    
    track_timeout_seconds = temporal_config.get(
        "track_timeout_seconds",
        2.0,
    )

    # --------------------------------------------------
    # Load YOLO Pose NCNN model
    # --------------------------------------------------

    print()
    print("Loading pose model...")
    print(f"Model: {args.model}")

    model = YOLO(
        args.model,
        task="pose",
    )

    # --------------------------------------------------
    # Temporal buffer
    # --------------------------------------------------

    buffer = TemporalPoseBuffer(
        maxlen=buffer_size
    )

    # --------------------------------------------------
    # Activity classifier
    # --------------------------------------------------

    classifier = ActivityClassifier(

        bending_angle_threshold=
            activity_config.get(
                "bending_angle_threshold",
                35.0,
            ),

        walking_velocity_threshold=
            activity_config.get(
                "walking_velocity_threshold",
                0.08,
            ),

        walking_ankle_threshold=
            activity_config.get(
                "walking_ankle_threshold",
                0.06,
            ),

        carrying_wrist_hip_threshold=
            activity_config.get(
                "carrying_wrist_hip_threshold",
                0.50,
            ),

        carrying_max_torso_angle=
            activity_config.get(
                "carrying_max_torso_angle",
                20.0,
            ),

        carrying_min_velocity=
            activity_config.get(
                "carrying_min_velocity",
                0.05,
            ),

        carrying_max_wrist_motion=
            activity_config.get(
                "carrying_max_wrist_motion",
                0.18,
            ),

        idle_velocity_threshold=
            activity_config.get(
                "idle_velocity_threshold",
                0.015,
            ),

        idle_ankle_threshold=
            activity_config.get(
                "idle_ankle_threshold",
                0.02,
            ),

        idle_min_frames=
            activity_config.get(
                "idle_min_frames",
                15,
            ),

        recent_window_size=
            activity_config.get(
                "recent_window_size",
                10,
            ),
    )

    # --------------------------------------------------
    # State smoother
    # --------------------------------------------------

    smoother = StateSmoother(
        window_size=smoothing_config.get(
            "window_size",
            5,
        ),
        required_votes=smoothing_config.get(
            "required_votes",
            3,
        ),
        transition_frames=smoothing_config.get(
            "transition_frames",
            3,
        ),
    )

    # --------------------------------------------------
    # Material-handling detector
    # --------------------------------------------------

    material_enabled = material_config.get(
        "enabled",
        True,
    )

    material_detector = MaterialHandlingDetector(
        window_size=material_config.get(
            "window_size",
            33,
        ),
        min_frames=material_config.get(
            "min_frames",
            20,
        ),
        min_wrist_motion=material_config.get(
            "min_wrist_motion",
            0.10,
        ),
        min_bending_ratio=material_config.get(
            "min_bending_ratio",
            0.12,
        ),
        min_active_state_ratio=material_config.get(
            "min_active_state_ratio",
            0.35,
        ),
        max_walking_ratio=material_config.get(
            "max_walking_ratio",
            0.75,
        ),
        min_completed_bending_cycles=material_config.get(
            "min_completed_bending_cycles",
            1,
        ),
        stationary_max_velocity=material_config.get(
            "stationary_max_velocity",
            0.05,
        ),
        stationary_min_wrist_motion=material_config.get(
            "stationary_min_wrist_motion",
            0.05,
        ),
        stationary_min_wrist_hip_distance=
            material_config.get(
                "stationary_min_wrist_hip_distance",
                0.50,
            ),
        stationary_min_standing_idle_ratio=
            material_config.get(
                "stationary_min_standing_idle_ratio",
                0.60,
            ),
        stationary_max_bending_ratio=material_config.get(
            "stationary_max_bending_ratio",
            0.10,
        ),
    )

    # --------------------------------------------------
    # Open video source
    # --------------------------------------------------

    print(f"Opening source: {args.source}")

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open source: {args.source}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 10.0

    print(
        f"Video: {width}x{height} @ {fps:.2f} FPS"
    )
    
    track_timeout_frames = max(
        1,
        int(track_timeout_seconds * fps),
    )

    # --------------------------------------------------
    # Output video
    # --------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Unable to create output video: "
            f"{output_path}"
        )

    # --------------------------------------------------
    # CSV logging
    # --------------------------------------------------

    csv_path = Path(args.csv)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "timestamp",
        "frame",
        "video_time_sec",
        "track_id",
        "raw_activity",
        "smoothed_activity",
        "semantic_activity",
        "material_handling",
        "material_confidence",
        "confidence",
        "velocity",
        "ankle_motion",
        "wrist_motion",
        "wrist_hip_distance",
        "torso_angle",
        "knee_angle",
        "mh_bending_ratio",
        "mh_walking_ratio",
        "mh_active_ratio",
        "mh_wrist_motion",
        "mh_pickup",
        "mh_stationary",
        "mh_standing_idle_ratio",
        "mh_wrist_hip_distance",
    ])

    # --------------------------------------------------
    # Processing variables
    # --------------------------------------------------

    last_seen = {}
    frame_number = 0
    start_time = time.time()
    
    def cleanup_stale_tracks():
        stale_track_ids = []

        for tracked_id, seen_frame in last_seen.items():

            frames_missing = (
                frame_number - seen_frame
            )

            if frames_missing > track_timeout_frames:
                stale_track_ids.append(tracked_id)

        for stale_id in stale_track_ids:

            buffer.remove(stale_id)
            smoother.remove(stale_id)
            material_detector.remove(stale_id)

            del last_seen[stale_id]

            print(
                f"Removed stale track ID={stale_id}"
            )

    print()
    print("Starting activity recognition...")
    print("Press Ctrl+C to stop.")
    print()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("End of stream/video.")
                break

            frame_number += 1

            # ------------------------------------------
            # YOLO Pose + tracking
            # ------------------------------------------

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=imgsz,
                conf=pose_conf,
                verbose=False,
            )

            if not results:
                cleanup_stale_tracks()
                writer.write(frame)
                continue

            result = results[0]

            # Draw skeleton + keypoints + boxes.
            annotated = result.plot()

            if result.boxes is None:
                cleanup_stale_tracks()
                writer.write(annotated)
                continue

            if result.keypoints is None:
                cleanup_stale_tracks()
                writer.write(annotated)
                continue

            if result.boxes.id is None:
                cleanup_stale_tracks()
                writer.write(annotated)
                continue

            # ------------------------------------------
            # Extract detections
            # ------------------------------------------

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            keypoints = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )
            
            keypoint_confidences = None

            if result.keypoints.conf is not None:
                keypoint_confidences = (
                    result.keypoints.conf
                    .cpu()
                    .numpy()
                )

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            # ------------------------------------------
            # Process every tracked person
            # ------------------------------------------

            if keypoint_confidences is None:
                keypoint_confidences = [
                    None
                    for _ in range(len(keypoints))
                ]

            for (
                box,
                person_keypoints,
                person_keypoint_confidences,
                track_id,
            ) in zip(
                boxes,
                keypoints,
                keypoint_confidences,
                track_ids,
            ):

                # Remember the most recent frame
                # where this track was detected.
                last_seen[track_id] = frame_number

                features = extract_pose_features(
                    person_keypoints,
                    confidences=person_keypoint_confidences,
                    min_confidence=keypoint_conf,
                )

                if features is None:
                    continue

                # Add pose features to history.
                buffer.add(
                    track_id,
                    features,
                )

                window = buffer.get(
                    track_id
                )

                # --------------------------------------
                # Raw activity classification
                # --------------------------------------

                state = classifier.classify(
                    window
                )

                raw_activity = state.get(
                    "activity",
                    "unknown",
                )

                # --------------------------------------
                # Smooth the activity state
                # --------------------------------------

                confidence = state.get(
                    "confidence",
                    0.0,
                )

                smoothed_activity = smoother.update(
                    track_id,
                    raw_activity,
                    confidence,
                )

                velocity = state.get(
                    "velocity",
                    0.0,
                )

                ankle_motion = state.get(
                    "ankle_motion",
                    0.0,
                )

                wrist_motion = state.get(
                    "wrist_motion",
                    0.0,
                )

                wrist_hip_distance = state.get(
                    "wrist_hip_distance"
                )

                torso_angle = state.get(
                    "torso_angle",
                    0.0,
                )

                knee_angle = state.get(
                    "knee_angle"
                )

                # --------------------------------------
                # Higher-level semantic activity
                # --------------------------------------

                if material_enabled:
                    material_state = material_detector.update(
                        track_id=track_id,
                        activity=raw_activity,
                        wrist_motion=wrist_motion,
                        wrist_hip_distance=wrist_hip_distance,
                        torso_angle=torso_angle,
                        velocity=velocity,
                    )
                else:
                    material_state = {
                        "detected": False,
                        "pickup_detected": False,
                        "stationary_detected": False,
                        "confidence": 0.0,
                        "bending_ratio": 0.0,
                        "walking_ratio": 0.0,
                        "active_state_ratio": 0.0,
                        "standing_idle_ratio": 0.0,
                        "mean_wrist_motion": 0.0,
                        "mean_wrist_hip_distance": 0.0,
                        "completed_bending_cycles": 0,
                    }

                material_handling = material_state[
                    "detected"
                ]

                material_confidence = material_state[
                    "confidence"
                ]

                if material_handling:
                    semantic_activity = "material_handling"
                else:
                    semantic_activity = smoothed_activity

                display_activity = semantic_activity

                # --------------------------------------
                # Draw semantic activity label
                # --------------------------------------

                draw_activity(
                    annotated,
                    box,
                    track_id,
                    display_activity,
                    (
                        material_confidence
                        if material_handling
                        else confidence
                    ),
                )

                # --------------------------------------
                # Terminal debug output
                # --------------------------------------

                knee_text = (
                    f"{knee_angle:.1f}"
                    if knee_angle is not None
                    else "None"
                )

                wrist_hip_text = (
                    f"{wrist_hip_distance:.3f}"
                    if wrist_hip_distance is not None
                    else "None"
                )

                print(
                    f"frame={frame_number} "
                    f"ID={track_id} "
                    f"raw={raw_activity} "
                    f"smooth={smoothed_activity} "
                    f"semantic={semantic_activity} "
                    f"mh={int(material_handling)} "
                    f"mh_conf={material_confidence:.2f} "
                    f"conf={confidence:.2f} "
                    f"velocity={velocity:.3f} "
                    f"ankle={ankle_motion:.3f} "
                    f"wrist={wrist_motion:.3f} "
                    f"wrist_hip={wrist_hip_text} "
                    f"torso={torso_angle:.1f} "
                    f"knee={knee_text}"
                )

                # --------------------------------------
                # CSV logging
                # --------------------------------------

                csv_writer.writerow([
                    time.time(),
                    frame_number,
                    (frame_number - 1) / fps,
                    track_id,
                    raw_activity,
                    smoothed_activity,
                    semantic_activity,
                    int(material_handling),
                    material_confidence,
                    confidence,
                    velocity,
                    ankle_motion,
                    wrist_motion,
                    wrist_hip_distance,
                    torso_angle,
                    knee_angle,
                    material_state["bending_ratio"],
                    material_state["walking_ratio"],
                    material_state["active_state_ratio"],
                    material_state["mean_wrist_motion"],
                    int(material_state["pickup_detected"]),
                    int(material_state["stationary_detected"]),
                    material_state["standing_idle_ratio"],
                    material_state["mean_wrist_hip_distance"],
                    material_state["completed_bending_cycles"],
                ])
                
            # ------------------------------------------
            # Remove stale track histories
            # ------------------------------------------

            cleanup_stale_tracks()

            # ------------------------------------------
            # Save annotated frame
            # ------------------------------------------

            writer.write(
                annotated
            )

            # ------------------------------------------
            # Optional live display
            # ------------------------------------------

            if args.show:

                cv2.imshow(
                    "Pose Activity Recognition",
                    annotated,
                )

                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):
                    print(
                        "User requested stop."
                    )
                    break

    except KeyboardInterrupt:

        print()
        print("Stopped by user.")

    finally:

        cap.release()
        writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

    # --------------------------------------------------
    # Final statistics
    # --------------------------------------------------

    elapsed = time.time() - start_time

    processing_fps = (
        frame_number / elapsed
        if elapsed > 0
        else 0.0
    )

    print()
    print("Finished.")
    print(
        f"Frames processed: {frame_number}"
    )
    print(
        f"Processing time: {elapsed:.2f} seconds"
    )
    print(
        f"Average processing FPS: "
        f"{processing_fps:.2f}"
    )
    print(
        f"Video saved to: {output_path}"
    )
    print(
        f"CSV saved to: {csv_path}"
    )


if __name__ == "__main__":
    main()
