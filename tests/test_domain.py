from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from bloodsmear.domain import (
    Detection,
    ImageInfo,
    InferenceOptions,
    InferenceResult,
    ModelInfo,
    QCInfo,
    TimingInfo,
    build_summary,
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


def detection(class_id: int, confidence: float = 0.9) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=CLASSES[class_id],
        confidence=confidence,
        bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
    )


def test_build_summary_includes_rbc_in_total_and_all_cell_ratio() -> None:
    detections = [detection(6), detection(6), detection(5), detection(4)]

    summary, warnings = build_summary(detections, CLASSES)

    assert summary.cell_counts["RBC"] == 2
    assert summary.total_detected_cells == 4
    assert summary.all_cell_ratios["RBC"] == pytest.approx(0.5)
    assert warnings == []


def test_build_summary_excludes_rbc_and_platelets_from_wbc_denominator() -> None:
    detections = [
        detection(4),
        detection(4),
        detection(2),
        detection(5),
        detection(6),
        detection(6),
    ]

    summary, _ = build_summary(detections, CLASSES)

    assert summary.wbc_differential_ratios["Neutrophil"] == pytest.approx(2 / 3)
    assert summary.wbc_differential_ratios["Lymphocyte"] == pytest.approx(1 / 3)
    assert "RBC" not in summary.wbc_differential_ratios
    assert "Platelets" not in summary.wbc_differential_ratios


def test_build_summary_returns_null_wbc_ratios_when_no_wbc() -> None:
    summary, warnings = build_summary([detection(6), detection(5)], CLASSES)

    assert all(value is None for value in summary.wbc_differential_ratios.values())
    assert "NO_WBC_DETECTED" in warnings


def test_build_summary_returns_null_all_cell_ratios_when_empty() -> None:
    summary, warnings = build_summary([], CLASSES)

    assert summary.total_detected_cells == 0
    assert all(value is None for value in summary.all_cell_ratios.values())
    assert "NO_WBC_DETECTED" in warnings


def test_build_summary_rejects_class_name_id_mismatch() -> None:
    mismatched = Detection(
        class_id=0,
        class_name="RBC",
        confidence=0.8,
        bbox_xyxy=(1.0, 1.0, 2.0, 2.0),
    )

    with pytest.raises(ValueError, match="class mapping"):
        build_summary([mismatched], CLASSES)


@pytest.mark.parametrize(
    "bbox",
    [(3.0, 2.0, 1.0, 4.0), (1.0, 5.0, 2.0, 4.0)],
)
def test_detection_rejects_reversed_coordinates(
    bbox: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValidationError):
        Detection(
            class_id=0,
            class_name="Basophil",
            confidence=0.8,
            bbox_xyxy=bbox,
        )


def test_detection_rejects_confidence_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Detection(
            class_id=0,
            class_name="Basophil",
            confidence=1.01,
            bbox_xyxy=(1.0, 1.0, 2.0, 2.0),
        )


def test_inference_options_use_approved_defaults() -> None:
    options = InferenceOptions()

    assert options.confidence_threshold == 0.25
    assert options.iou_threshold == 0.70
    assert options.max_detections == 3000
    assert options.require_gpu is False


def test_inference_result_serializes_typed_contract() -> None:
    summary, warnings = build_summary([detection(4)], CLASSES)
    result = InferenceResult(
        sample_id="S001",
        created_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
        model=ModelInfo(name="blood-cell-yolo11", version="1.0.0", sha256="A" * 64),
        image=ImageInfo(
            filename="sample.png",
            width=100,
            height=80,
            sha256="B" * 64,
        ),
        detections=[detection(4)],
        summary=summary,
        qc=QCInfo(status="pass", warnings=warnings),
        runtime=TimingInfo(
            provider="CPUExecutionProvider",
            preprocess_ms=1.0,
            inference_ms=2.0,
            postprocess_ms=1.0,
            total_ms=4.0,
        ),
    )

    payload = result.model_dump(mode="json")

    assert payload["schema_version"] == "1.0"
    assert payload["summary"]["cell_counts"]["Neutrophil"] == 1
    assert payload["runtime"]["total_ms"] == 4.0
