#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


PPE_ITEMS = ["helmet", "vest", "gloves", "boots"]


def load_ground_truth(path):
    rows = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "activity": row["activity"],
                    "helmet": int(row["helmet"]),
                    "vest": int(row["vest"]),
                    "gloves": int(row["gloves"]),
                    "boots": int(row["boots"]),
                }
            )

    return rows


def ground_truth_at_time(gt_rows, timestamp):
    for row in gt_rows:
        if row["start_sec"] <= timestamp < row["end_sec"]:
            return row

    return None


def safe_div(num, den):
    return num / den if den else 0.0


def calculate_metrics(tp, tn, fp, fn):
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def evaluate(prediction_path, gt_rows, backend_name):
    confusion = {
        item: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for item in PPE_ITEMS
    }

    labelled_frames = 0
    unlabelled_frames = 0

    error_rows = []

    with open(prediction_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame = int(row["frame"])
            timestamp = float(row["time_sec"])

            gt = ground_truth_at_time(gt_rows, timestamp)

            if gt is None:
                unlabelled_frames += 1
                continue

            labelled_frames += 1

            for item in PPE_ITEMS:
                actual = gt[item]
                predicted = int(row[f"{item}_present"])

                if actual == 1 and predicted == 1:
                    confusion[item]["tp"] += 1

                elif actual == 0 and predicted == 0:
                    confusion[item]["tn"] += 1

                elif actual == 0 and predicted == 1:
                    confusion[item]["fp"] += 1

                    error_rows.append(
                        {
                            "backend": backend_name,
                            "frame": frame,
                            "time_sec": timestamp,
                            "activity": gt["activity"],
                            "ppe_item": item,
                            "ground_truth": actual,
                            "prediction": predicted,
                            "error": "FP",
                            "confidence": row[f"{item}_conf"],
                        }
                    )

                elif actual == 1 and predicted == 0:
                    confusion[item]["fn"] += 1

                    error_rows.append(
                        {
                            "backend": backend_name,
                            "frame": frame,
                            "time_sec": timestamp,
                            "activity": gt["activity"],
                            "ppe_item": item,
                            "ground_truth": actual,
                            "prediction": predicted,
                            "error": "FN",
                            "confidence": row[f"{item}_conf"],
                        }
                    )

    results = []

    for item in PPE_ITEMS:
        c = confusion[item]

        metrics = calculate_metrics(
            c["tp"],
            c["tn"],
            c["fp"],
            c["fn"],
        )

        results.append(
            {
                "backend": backend_name,
                "ppe_item": item,
                "labelled_frames": labelled_frames,
                "tp": c["tp"],
                "tn": c["tn"],
                "fp": c["fp"],
                "fn": c["fn"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
            }
        )

    return results, error_rows, labelled_frames, unlabelled_frames


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--pytorch", required=True)
    parser.add_argument("--ncnn", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    gt_rows = load_ground_truth(args.ground_truth)

    output_dir = Path(args.output_dir)

    pt_results, pt_errors, pt_labelled, pt_unlabelled = evaluate(
        args.pytorch,
        gt_rows,
        "PyTorch",
    )

    ncnn_results, ncnn_errors, ncnn_labelled, ncnn_unlabelled = evaluate(
        args.ncnn,
        gt_rows,
        "NCNN",
    )

    results = pt_results + ncnn_results
    errors = pt_errors + ncnn_errors

    metrics_file = output_dir / "ppe_ground_truth_metrics.csv"
    errors_file = output_dir / "ppe_ground_truth_errors.csv"

    metric_fields = [
        "backend",
        "ppe_item",
        "labelled_frames",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]

    error_fields = [
        "backend",
        "frame",
        "time_sec",
        "activity",
        "ppe_item",
        "ground_truth",
        "prediction",
        "error",
        "confidence",
    ]

    write_csv(metrics_file, results, metric_fields)
    write_csv(errors_file, errors, error_fields)

    print()
    print("PPE FRAME-LEVEL GROUND-TRUTH EVALUATION")
    print("=" * 90)

    for result in results:
        print(
            f"{result['backend']:8s} | "
            f"{result['ppe_item']:7s} | "
            f"Acc={result['accuracy']:.3f} | "
            f"Prec={result['precision']:.3f} | "
            f"Recall={result['recall']:.3f} | "
            f"F1={result['f1']:.3f} | "
            f"TP={result['tp']} "
            f"TN={result['tn']} "
            f"FP={result['fp']} "
            f"FN={result['fn']}"
        )

    print()
    print(f"PyTorch labelled frames:   {pt_labelled}")
    print(f"PyTorch excluded frames:   {pt_unlabelled}")
    print(f"NCNN labelled frames:      {ncnn_labelled}")
    print(f"NCNN excluded frames:      {ncnn_unlabelled}")

    print()
    print(f"Metrics saved to: {metrics_file}")
    print(f"Errors saved to:  {errors_file}")


if __name__ == "__main__":
    main()
