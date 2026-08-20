"""PPE detection mapping and conservative person-track association."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence


PPE_ITEMS = ("helmet", "vest", "gloves", "boots")
NEGATIVE_CLASSES = {
    "helmet": "no_helmet",
    "gloves": "no_gloves",
    "boots": "no_boots",
}


def _intersection_over_detection(detection, person) -> float:
    dx1, dy1, dx2, dy2 = map(float, detection)
    px1, py1, px2, py2 = map(float, person)
    intersection = max(0.0, min(dx2, px2) - max(dx1, px1)) * max(
        0.0, min(dy2, py2) - max(dy1, py1)
    )
    area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
    return intersection / area if area else 0.0


def associate_ppe(
    person_boxes: Sequence[Sequence[float]],
    detections: Iterable[dict],
    observed_at: str | None = None,
    containment_threshold: float = 0.5,
) -> list[dict]:
    """Associate PPE boxes to exactly one pose person.

    A detection must have its centre inside a person box or at least the configured
    fraction of its area contained by it. Detections matching multiple people are
    discarded as ambiguous. Missing classes remain unknown; only explicit model
    negative classes produce ``detected=False``.
    """

    timestamp = observed_at or datetime.now().astimezone().isoformat()
    matches: list[list[dict]] = [[] for _ in person_boxes]
    for detection in detections:
        box = detection["box"]
        x1, y1, x2, y2 = map(float, box)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        candidates = []
        for index, person in enumerate(person_boxes):
            px1, py1, px2, py2 = map(float, person)
            center_inside = px1 <= center[0] <= px2 and py1 <= center[1] <= py2
            contained = _intersection_over_detection(box, person)
            if center_inside or contained >= containment_threshold:
                candidates.append((index, contained))
        if len(candidates) == 1:
            index, score = candidates[0]
            matches[index].append({**detection, "association_score": score})

    states = []
    for associated in matches:
        items = {}
        for item in PPE_ITEMS:
            positive = [d for d in associated if d["class_name"] == item]
            negative_name = NEGATIVE_CLASSES.get(item)
            negative = [d for d in associated if d["class_name"] == negative_name]
            candidates = [(True, d) for d in positive] + [(False, d) for d in negative]
            if not candidates:
                items[item] = {"detected": None, "confidence": None}
                continue
            detected, best = max(candidates, key=lambda value: float(value[1]["confidence"]))
            items[item] = {
                "detected": detected,
                "confidence": float(best["confidence"]),
            }
        states.append({
            **items,
            "observed_at": timestamp,
            "association_method": "ppe_box_within_pose_person",
        })
    return states


def result_detections(result) -> list[dict]:
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    return [
        {
            "class_name": result.names[int(class_id)],
            "confidence": float(confidence),
            "box": box.tolist(),
        }
        for box, class_id, confidence in zip(boxes, classes, confidences)
    ]
