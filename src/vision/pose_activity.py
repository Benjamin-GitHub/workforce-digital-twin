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

    imgsz = pose_config.get("imgsz", 320)
    pose_conf = pose_config.get("confidence", 0.35)

    buffer_size = temporal_config.get(
        "buffer_size",
        20,
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
        "track_id",
        "raw_activity",
        "smoothed_activity",
        "confidence",
        "velocity",
        "ankle_motion",
        "torso_angle",
        "knee_angle",
    ])

    # --------------------------------------------------
    # Processing variables
    # --------------------------------------------------

    frame_number = 0
    start_time = time.time()

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
                writer.write(frame)
                continue

            result = results[0]

            # Draw skeleton + keypoints + boxes.
            annotated = result.plot()

            if result.boxes is None:
                writer.write(annotated)
                continue

            if result.keypoints is None:
                writer.write(annotated)
                continue

            if result.boxes.id is None:
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

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            # ------------------------------------------
            # Process every tracked person
            # ------------------------------------------

            for (
                box,
                person_keypoints,
                track_id,
            ) in zip(
                boxes,
                keypoints,
                track_ids,
            ):

                features = extract_pose_features(
                    person_keypoints
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

                smoothed_activity = smoother.update(
                    track_id,
                    raw_activity,
                )

                confidence = state.get(
                    "confidence",
                    0.0,
                )

                velocity = state.get(
                    "velocity",
                    0.0,
                )

                ankle_motion = state.get(
                    "ankle_motion",
                    0.0,
                )

                torso_angle = state.get(
                    "torso_angle",
                    0.0,
                )

                knee_angle = state.get(
                    "knee_angle"
                )

                # --------------------------------------
                # Draw SMOOTHED activity label
                # --------------------------------------

                draw_activity(
                    annotated,
                    box,
                    track_id,
                    smoothed_activity,
                    confidence,
                )

                # --------------------------------------
                # Terminal debug output
                # --------------------------------------

                knee_text = (
                    f"{knee_angle:.1f}"
                    if knee_angle is not None
                    else "None"
                )

                print(
                    f"frame={frame_number} "
                    f"ID={track_id} "
                    f"raw={raw_activity} "
                    f"smooth={smoothed_activity} "
                    f"conf={confidence:.2f} "
                    f"velocity={velocity:.3f} "
                    f"ankle={ankle_motion:.3f} "
                    f"torso={torso_angle:.1f} "
                    f"knee={knee_text}"
                )

                # --------------------------------------
                # CSV logging
                # --------------------------------------

                csv_writer.writerow([
                    time.time(),
                    frame_number,
                    track_id,
                    raw_activity,
                    smoothed_activity,
                    confidence,
                    velocity,
                    ankle_motion,
                    torso_angle,
                    knee_angle,
                ])

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
