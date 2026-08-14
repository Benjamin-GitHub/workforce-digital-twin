#!/usr/bin/env python3
"""
Evaluate the frozen six-class pose/activity pipeline against the clean,
single-activity ground-truth intervals from the controlled 5-minute video.

Default evaluation target:
    semantic_activity

Outputs:
    activity_frame_results.csv
    activity_segment_results.csv
    activity_classification_report.csv
    activity_confusion_matrix.csv
    activity_confusion_matrix.png
    activity_per_class_f1.png
    activity_timeline.png
    activity_evaluation_summary.txt

The primary evaluation:
- uses only clean, single-activity ground-truth intervals;
- treats intervals as half-open [start_sec, end_sec);
- evaluates one primary tracked worker (default track_id=1);
- counts a missing primary-track prediction as incorrect rather than dropping it;
- reports coverage separately;
- optionally reports boundary-tolerant accuracy.

No scikit-learn dependency is required.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


CLASSES: Tuple[str, ...] = (
    "walking",
    "standing",
    "idle",
    "bending",
    "carrying",
    "material_handling",
)

MISSING_LABEL = "__missing__"


@dataclass(frozen=True)
class GroundTruthSegment:
    label: str
    start_sec: float
    end_sec: float
    description: str


# Clean single-activity ground truth only.
#
# Composite/mixed intervals from the controlled video are intentionally excluded:
# - Bending - Short walk
# - Standing / material handling
# - Walking / standing
# - Standing / Walking
# - Bending / Carrying
# - Walking / Standing / Walking
# - Mixed natural sequence
#
# The 01:46-02:00 region is included as material_handling based on the
# clarified ground truth: the worker is standing while holding/working
# with material.
GROUND_TRUTH_SEGMENTS: Tuple[GroundTruthSegment, ...] = (
    GroundTruthSegment("walking", 0.0, 5.0, "Walking"),
    GroundTruthSegment("idle", 5.0, 20.0, "Idle"),
    GroundTruthSegment("walking", 21.0, 40.0, "Walking"),
    GroundTruthSegment(
        "material_handling",
        60.0,
        88.0,
        "Material handling - bending/picking up boxes",
    ),
    GroundTruthSegment("carrying", 89.0, 105.0, "Carrying"),
    GroundTruthSegment(
        "material_handling",
        106.0,
        120.0,
        "Material handling - standing/manual handling",
    ),
    GroundTruthSegment("standing", 125.0, 133.0, "Standing"),
    GroundTruthSegment("standing", 185.0, 190.0, "Standing"),
    GroundTruthSegment("walking", 222.0, 233.0, "Walking"),
    GroundTruthSegment(
        "material_handling",
        233.0,
        240.0,
        "Material handling",
    ),
    GroundTruthSegment("bending", 242.0, 243.0, "Bending"),
    GroundTruthSegment("standing", 243.0, 275.0, "Standing"),
    GroundTruthSegment("idle", 276.0, 288.0, "Idle"),
    GroundTruthSegment("walking", 288.0, 300.0, "Walking"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate activity predictions against clean controlled-video "
            "ground truth."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Input activity CSV produced by pose_activity.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for evaluation CSVs, plots and summary.",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=1,
        help="Primary worker track ID to evaluate (default: 1).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=11.0,
        help="Canonical processed-video FPS (default: 11).",
    )
    parser.add_argument(
        "--prediction-column",
        default="semantic_activity",
        choices=("semantic_activity", "smoothed_activity", "raw_activity"),
        help="Prediction column to evaluate (default: semantic_activity).",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=0.5,
        help=(
            "Seconds to exclude from each labelled segment edge for the "
            "boundary-tolerant accuracy (default: 0.5)."
        ),
    )
    return parser.parse_args()


def safe_float(value: str, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: str, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalise_label(value: Optional[str]) -> str:
    if value is None:
        return MISSING_LABEL

    label = value.strip().lower().replace(" ", "_")

    aliases = {
        "materialhandling": "material_handling",
        "material-handling": "material_handling",
    }
    label = aliases.get(label, label)

    if not label:
        return MISSING_LABEL

    return label


def read_primary_predictions(
    path: Path,
    track_id: int,
    prediction_column: str,
) -> Tuple[Dict[int, dict], List[str]]:
    """Read one prediction row per frame for the requested primary track."""
    predictions: Dict[int, dict] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")

        required = {"frame", "track_id", prediction_column}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Input CSV is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        fieldnames = list(reader.fieldnames)

        for row in reader:
            row_track_id = safe_int(row.get("track_id", ""))
            if row_track_id != track_id:
                continue

            frame = safe_int(row.get("frame", ""))
            if frame < 1:
                continue

            # If a duplicate row somehow exists for the same track/frame,
            # retain the last one deterministically.
            predictions[frame] = row

    return predictions, fieldnames


def gt_segment_for_time(
    time_sec: float,
) -> Optional[Tuple[int, GroundTruthSegment]]:
    for index, segment in enumerate(GROUND_TRUTH_SEGMENTS, start=1):
        if segment.start_sec <= time_sec < segment.end_sec:
            return index, segment
    return None


def expected_gt_frames(fps: float) -> List[Tuple[int, float, int, GroundTruthSegment]]:
    """
    Construct labelled ground-truth frames using frame 1 at t=0 and
    t=(frame-1)/fps. Intervals are half-open [start,end).
    """
    if fps <= 0:
        raise ValueError("FPS must be positive.")

    max_end = max(segment.end_sec for segment in GROUND_TRUTH_SEGMENTS)
    max_frame = int(math.floor(max_end * fps)) + 1

    rows: List[Tuple[int, float, int, GroundTruthSegment]] = []

    for frame in range(1, max_frame + 1):
        time_sec = (frame - 1) / fps
        match = gt_segment_for_time(time_sec)
        if match is None:
            continue

        segment_index, segment = match
        rows.append((frame, time_sec, segment_index, segment))

    return rows


def confusion_counts(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = {
        true_label: Counter()
        for true_label in labels
    }

    for true_label, pred_label in zip(y_true, y_pred):
        counts[true_label][pred_label] += 1

    return counts


def calculate_class_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> List[dict]:
    rows: List[dict] = []

    for label in labels:
        tp = sum(
            1
            for true_label, pred_label in zip(y_true, y_pred)
            if true_label == label and pred_label == label
        )
        fp = sum(
            1
            for true_label, pred_label in zip(y_true, y_pred)
            if true_label != label and pred_label == label
        )
        fn = sum(
            1
            for true_label, pred_label in zip(y_true, y_pred)
            if true_label == label and pred_label != label
        )
        support = sum(1 for true_label in y_true if true_label == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        rows.append({
            "class": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    return rows


def strict_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0

    correct = sum(
        true_label == pred_label
        for true_label, pred_label in zip(y_true, y_pred)
    )
    return correct / len(y_true)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(
    path: Path,
    matrix: np.ndarray,
    labels: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest", aspect="auto")
    fig.colorbar(image, ax=ax)

    display_labels = [label.replace("_", " ").title() for label in labels]
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
        xlabel="Predicted activity",
        ylabel="Ground-truth activity",
        title="Activity Recognition Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(path: Path, metrics: Sequence[dict]) -> None:
    labels = [row["class"].replace("_", " ").title() for row in metrics]
    values = [row["f1"] for row in metrics]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-Class Activity F1 Score")
    ax.tick_params(axis="x", rotation=35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 0.98),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(
    path: Path,
    frame_rows: Sequence[dict],
    labels: Sequence[str],
) -> None:
    class_to_y = {label: index for index, label in enumerate(labels)}
    unknown_y = len(labels)

    times = [float(row["video_time_sec"]) for row in frame_rows]
    gt_values = [class_to_y[row["ground_truth"]] for row in frame_rows]
    pred_values = [
        class_to_y.get(row["prediction"], unknown_y)
        for row in frame_rows
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(times, gt_values, s=8, marker="s", label="Ground truth")
    ax.scatter(times, pred_values, s=8, marker=".", label="Prediction", alpha=0.65)

    ylabels = [label.replace("_", " ").title() for label in labels] + [
        "Missing / Other"
    ]
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Video time (s)")
    ax.set_ylabel("Activity")
    ax.set_title("Ground Truth vs Predicted Activity Timeline")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions, input_fieldnames = read_primary_predictions(
        args.csv,
        args.track_id,
        args.prediction_column,
    )

    gt_frames = expected_gt_frames(args.fps)

    frame_rows: List[dict] = []
    y_true: List[str] = []
    y_pred: List[str] = []

    missing_primary_frames = 0

    for frame, time_sec, segment_index, segment in gt_frames:
        row = predictions.get(frame)

        if row is None:
            prediction = MISSING_LABEL
            missing_primary_frames += 1
            raw_activity = ""
            smoothed_activity = ""
            semantic_activity = ""
        else:
            prediction = normalise_label(row.get(args.prediction_column))
            raw_activity = normalise_label(row.get("raw_activity"))
            smoothed_activity = normalise_label(row.get("smoothed_activity"))
            semantic_activity = normalise_label(row.get("semantic_activity"))

        y_true.append(segment.label)
        y_pred.append(prediction)

        frame_rows.append({
            "frame": frame,
            "video_time_sec": f"{time_sec:.6f}",
            "segment_id": segment_index,
            "segment_description": segment.description,
            "ground_truth": segment.label,
            "prediction": prediction,
            "correct": int(prediction == segment.label),
            "primary_track_present": int(row is not None),
            "raw_activity": raw_activity,
            "smoothed_activity": smoothed_activity,
            "semantic_activity": semantic_activity,
        })

    write_csv(
        args.output_dir / "activity_frame_results.csv",
        (
            "frame",
            "video_time_sec",
            "segment_id",
            "segment_description",
            "ground_truth",
            "prediction",
            "correct",
            "primary_track_present",
            "raw_activity",
            "smoothed_activity",
            "semantic_activity",
        ),
        frame_rows,
    )

    metrics = calculate_class_metrics(y_true, y_pred, CLASSES)

    write_csv(
        args.output_dir / "activity_classification_report.csv",
        ("class", "precision", "recall", "f1", "support", "tp", "fp", "fn"),
        metrics,
    )

    # Confusion matrix contains the six target classes only.
    # Predictions outside the six classes (including missing/unknown) are
    # still counted as false negatives in the metrics, but are not given
    # an extra plotted target-class column.
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    class_index = {label: index for index, label in enumerate(CLASSES)}

    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label in class_index:
            matrix[class_index[true_label], class_index[pred_label]] += 1

    confusion_rows = []
    for true_label in CLASSES:
        row = {"ground_truth": true_label}
        true_index = class_index[true_label]
        for pred_label in CLASSES:
            row[pred_label] = int(matrix[true_index, class_index[pred_label]])
        row["missing_or_other"] = sum(
            1
            for truth, pred in zip(y_true, y_pred)
            if truth == true_label and pred not in class_index
        )
        confusion_rows.append(row)

    write_csv(
        args.output_dir / "activity_confusion_matrix.csv",
        ("ground_truth", *CLASSES, "missing_or_other"),
        confusion_rows,
    )

    plot_confusion_matrix(
        args.output_dir / "activity_confusion_matrix.png",
        matrix,
        CLASSES,
    )
    plot_per_class_f1(
        args.output_dir / "activity_per_class_f1.png",
        metrics,
    )
    plot_timeline(
        args.output_dir / "activity_timeline.png",
        frame_rows,
        CLASSES,
    )

    segment_rows: List[dict] = []

    for segment_index, segment in enumerate(GROUND_TRUTH_SEGMENTS, start=1):
        rows = [
            row
            for row in frame_rows
            if row["segment_id"] == segment_index
        ]

        total = len(rows)
        correct = sum(int(row["correct"]) for row in rows)
        missing = sum(
            1
            for row in rows
            if int(row["primary_track_present"]) == 0
        )

        segment_rows.append({
            "segment_id": segment_index,
            "description": segment.description,
            "ground_truth": segment.label,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "duration_sec": segment.end_sec - segment.start_sec,
            "frames": total,
            "correct_frames": correct,
            "accuracy": correct / total if total else 0.0,
            "missing_primary_track_frames": missing,
        })

    write_csv(
        args.output_dir / "activity_segment_results.csv",
        (
            "segment_id",
            "description",
            "ground_truth",
            "start_sec",
            "end_sec",
            "duration_sec",
            "frames",
            "correct_frames",
            "accuracy",
            "missing_primary_track_frames",
        ),
        segment_rows,
    )

    strict_acc = strict_accuracy(y_true, y_pred)

    recalls = [row["recall"] for row in metrics]
    f1_values = [row["f1"] for row in metrics]

    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    coverage = (
        (len(y_true) - missing_primary_frames) / len(y_true)
        if y_true
        else 0.0
    )

    tolerance = max(0.0, args.boundary_tolerance)
    tolerant_pairs: List[Tuple[str, str]] = []

    for row in frame_rows:
        segment = GROUND_TRUTH_SEGMENTS[int(row["segment_id"]) - 1]
        time_sec = float(row["video_time_sec"])

        # Remove tolerance from both ends of every labelled segment.
        # Very short segments can disappear entirely; that is intentional
        # and is reported in the summary.
        if (
            time_sec >= segment.start_sec + tolerance
            and time_sec < segment.end_sec - tolerance
        ):
            tolerant_pairs.append(
                (row["ground_truth"], row["prediction"])
            )

    if tolerant_pairs:
        tolerant_true = [pair[0] for pair in tolerant_pairs]
        tolerant_pred = [pair[1] for pair in tolerant_pairs]
        tolerant_accuracy = strict_accuracy(tolerant_true, tolerant_pred)
    else:
        tolerant_accuracy = 0.0

    summary_lines = [
        "ACTIVITY RECOGNITION EVALUATION SUMMARY",
        "=" * 72,
        "",
        f"Input CSV: {args.csv}",
        f"Prediction column: {args.prediction_column}",
        f"Primary track ID: {args.track_id}",
        f"FPS: {args.fps:.3f}",
        "",
        "Evaluation protocol",
        "-" * 72,
        "Clean single-activity intervals only.",
        "Ground-truth intervals are half-open [start_sec, end_sec).",
        "Frame 1 is t=0; frame time = (frame - 1) / FPS.",
        "Missing primary-track frames are counted as incorrect.",
        "Composite/mixed ground-truth intervals are excluded.",
        "",
        "Overall results",
        "-" * 72,
        f"Evaluated GT frames: {len(y_true)}",
        f"Missing primary-track frames: {missing_primary_frames}",
        f"Primary-track coverage: {coverage * 100:.2f}%",
        f"Strict frame accuracy: {strict_acc * 100:.2f}%",
        f"Macro-F1: {macro_f1 * 100:.2f}%",
        f"Balanced accuracy: {balanced_accuracy * 100:.2f}%",
        (
            f"Boundary-tolerant accuracy (±{tolerance:.2f}s excluded): "
            f"{tolerant_accuracy * 100:.2f}%"
        ),
        f"Boundary-tolerant evaluated frames: {len(tolerant_pairs)}",
        "",
        "Per-class results",
        "-" * 72,
    ]

    for row in metrics:
        summary_lines.append(
            f"{row['class']:<20} "
            f"P={row['precision'] * 100:6.2f}%  "
            f"R={row['recall'] * 100:6.2f}%  "
            f"F1={row['f1'] * 100:6.2f}%  "
            f"support={row['support']}"
        )

    summary_lines.extend([
        "",
        "Notes",
        "-" * 72,
        (
            "Bending has very limited clean support (1 second in the current "
            "ground truth), so its metric should be interpreted cautiously."
        ),
        (
            "Material handling is evaluated as the final semantic activity; "
            "the lower-level raw/smoothed activities remain available in the "
            "frame-results CSV for diagnostic comparison."
        ),
        (
            "This script intentionally does not force composite/mixed intervals "
            "into a single ground-truth class."
        ),
        "",
    ])

    summary_path = args.output_dir / "activity_evaluation_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print("Generated files:")
    for name in (
        "activity_frame_results.csv",
        "activity_segment_results.csv",
        "activity_classification_report.csv",
        "activity_confusion_matrix.csv",
        "activity_confusion_matrix.png",
        "activity_per_class_f1.png",
        "activity_timeline.png",
        "activity_evaluation_summary.txt",
    ):
        print(f"  {args.output_dir / name}")


if __name__ == "__main__":
    main()
