from __future__ import annotations

import numpy as np
import pytest

from bloodsmear.domain import Detection
from bloodsmear.image_ops import LetterboxTransform
from bloodsmear.postprocess import (
    class_aware_nms,
    decode_yolo_output,
    normalize_yolo_output,
)


CLASSES = [
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelets",
    "RBC",
]


def transform() -> LetterboxTransform:
    return LetterboxTransform(
        original_width=640,
        original_height=640,
        input_size=640,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
    )


def prediction(
    *,
    cx: float = 100.0,
    cy: float = 100.0,
    width: float = 40.0,
    height: float = 40.0,
    class_id: int = 0,
    confidence: float = 0.9,
) -> np.ndarray:
    row = np.zeros(4 + len(CLASSES), dtype=np.float32)
    row[:4] = (cx, cy, width, height)
    row[4 + class_id] = confidence
    return row


def detection(
    class_id: int,
    confidence: float,
    bbox: tuple[float, float, float, float],
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=CLASSES[class_id],
        confidence=confidence,
        bbox_xyxy=bbox,
    )


def test_normalize_yolo_output_accepts_channels_first_layout() -> None:
    rows = np.stack([prediction(class_id=0), prediction(class_id=6)])
    output = rows.T[None, :, :]

    normalized = normalize_yolo_output(output, class_count=7)

    assert normalized.shape == (2, 11)
    assert normalized[1, 10] == pytest.approx(0.9)


def test_normalize_yolo_output_accepts_channels_last_layout() -> None:
    rows = np.stack([prediction(class_id=0), prediction(class_id=6)])

    normalized = normalize_yolo_output(rows[None, :, :], class_count=7)

    assert normalized.shape == (2, 11)


def test_normalize_yolo_output_rejects_wrong_feature_count() -> None:
    with pytest.raises(ValueError, match="feature dimension"):
        normalize_yolo_output(np.zeros((1, 10, 20), dtype=np.float32), 7)


def test_decode_yolo_output_filters_confidence_and_clips_coordinates() -> None:
    rows = np.stack(
        [
            prediction(cx=10, cy=10, width=40, height=40, class_id=4, confidence=0.9),
            prediction(class_id=2, confidence=0.2),
        ]
    )

    detections = decode_yolo_output(
        rows[None, :, :],
        transform(),
        CLASSES,
        confidence_threshold=0.25,
    )

    assert len(detections) == 1
    assert detections[0].class_name == "Neutrophil"
    assert detections[0].bbox_xyxy == pytest.approx((0.0, 0.0, 30.0, 30.0))


def test_decode_yolo_output_orders_by_confidence_deterministically() -> None:
    rows = np.stack(
        [
            prediction(class_id=1, confidence=0.7),
            prediction(class_id=6, confidence=0.95, cx=200),
        ]
    )

    detections = decode_yolo_output(rows[None, :, :], transform(), CLASSES, 0.25)

    assert [item.class_name for item in detections] == ["RBC", "Eosinophil"]


def test_class_aware_nms_suppresses_overlapping_same_class() -> None:
    detections = [
        detection(6, 0.9, (0.0, 0.0, 20.0, 20.0)),
        detection(6, 0.8, (1.0, 1.0, 21.0, 21.0)),
    ]

    kept, warnings = class_aware_nms(detections, iou_threshold=0.5, max_detections=10)

    assert kept == [detections[0]]
    assert warnings == []


def test_class_aware_nms_keeps_overlapping_different_classes() -> None:
    detections = [
        detection(6, 0.9, (0.0, 0.0, 20.0, 20.0)),
        detection(5, 0.8, (1.0, 1.0, 21.0, 21.0)),
    ]

    kept, _ = class_aware_nms(detections, iou_threshold=0.5, max_detections=10)

    assert kept == detections


def test_class_aware_nms_orders_results_by_descending_confidence() -> None:
    detections = [
        detection(0, 0.6, (0.0, 0.0, 10.0, 10.0)),
        detection(1, 0.9, (20.0, 0.0, 30.0, 10.0)),
        detection(2, 0.7, (40.0, 0.0, 50.0, 10.0)),
    ]

    kept, _ = class_aware_nms(detections, iou_threshold=0.5, max_detections=10)

    assert [item.confidence for item in kept] == [0.9, 0.7, 0.6]


def test_class_aware_nms_warns_when_exactly_maximum_is_retained() -> None:
    detections = [
        detection(index % 7, 0.9, (index * 20.0, 0.0, index * 20.0 + 10.0, 10.0))
        for index in range(3000)
    ]

    kept, warnings = class_aware_nms(
        detections,
        iou_threshold=0.5,
        max_detections=3000,
    )

    assert len(kept) == 3000
    assert "DETECTION_LIMIT_REACHED" in warnings
