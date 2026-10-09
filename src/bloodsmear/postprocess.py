from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from bloodsmear.domain import Detection
from bloodsmear.image_ops import LetterboxTransform


def normalize_yolo_output(
    output: NDArray[np.floating],
    class_count: int,
) -> NDArray[np.float32]:
    array = np.asarray(output)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"Expected YOLO output shape [1, *, *], got {array.shape}")

    matrix = array[0]
    feature_count = 4 + class_count
    if matrix.shape[0] == feature_count:
        matrix = matrix.T
    elif matrix.shape[1] != feature_count:
        raise ValueError(
            f"YOLO feature dimension must be {feature_count}, got {matrix.shape}"
        )
    return np.ascontiguousarray(matrix, dtype=np.float32)


def decode_yolo_output(
    output: NDArray[np.floating],
    transform: LetterboxTransform,
    classes: Sequence[str],
    confidence_threshold: float,
) -> list[Detection]:
    rows = normalize_yolo_output(output, class_count=len(classes))
    detections: list[Detection] = []
    for row in rows:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < confidence_threshold:
            continue

        center_x, center_y, width, height = (float(value) for value in row[:4])
        model_box = (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
        original_box = transform.to_original(model_box)
        if original_box[2] <= original_box[0] or original_box[3] <= original_box[1]:
            continue
        detections.append(
            Detection(
                class_id=class_id,
                class_name=classes[class_id],
                confidence=confidence,
                bbox_xyxy=original_box,
            )
        )

    return sorted(
        detections,
        key=lambda item: (
            -item.confidence,
            item.class_id,
            item.bbox_xyxy[0],
            item.bbox_xyxy[1],
        ),
    )


def _iou_one_to_many(
    box: NDArray[np.float32],
    boxes: NDArray[np.float32],
) -> NDArray[np.float32]:
    intersection_x1 = np.maximum(box[0], boxes[:, 0])
    intersection_y1 = np.maximum(box[1], boxes[:, 1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])
    intersection_width = np.maximum(0.0, intersection_x2 - intersection_x1)
    intersection_height = np.maximum(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height

    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def class_aware_nms(
    detections: Sequence[Detection],
    iou_threshold: float,
    max_detections: int,
) -> tuple[list[Detection], list[str]]:
    by_class: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_class[detection.class_id].append(detection)

    retained: list[Detection] = []
    for class_detections in by_class.values():
        ordered = sorted(
            class_detections,
            key=lambda item: (
                -item.confidence,
                item.bbox_xyxy[0],
                item.bbox_xyxy[1],
            ),
        )
        boxes = np.asarray(
            [item.bbox_xyxy for item in ordered],
            dtype=np.float32,
        )
        remaining = np.arange(len(ordered))
        while remaining.size:
            selected = int(remaining[0])
            retained.append(ordered[selected])
            if remaining.size == 1:
                break
            remaining_candidates = remaining[1:]
            overlaps = _iou_one_to_many(boxes[selected], boxes[remaining_candidates])
            remaining = remaining_candidates[overlaps <= iou_threshold]

    retained.sort(
        key=lambda item: (
            -item.confidence,
            item.class_id,
            item.bbox_xyxy[0],
            item.bbox_xyxy[1],
        )
    )
    retained = retained[:max_detections]
    warnings = ["DETECTION_LIMIT_REACHED"] if len(retained) == max_detections else []
    return retained, warnings
