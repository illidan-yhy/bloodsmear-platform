from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from bloodsmear.artifacts import (
    write_annotated_image,
    write_detections_csv,
    write_result_json,
)
from bloodsmear.domain import (
    Detection,
    ImageInfo,
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


def image_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (40, 20), color=(255, 255, 255)).save(stream, format="PNG")
    return stream.getvalue()


def result() -> InferenceResult:
    detections = [
        Detection(
            class_id=6,
            class_name="RBC",
            confidence=0.95,
            bbox_xyxy=(1.0, 2.0, 15.0, 18.0),
        ),
        Detection(
            class_id=4,
            class_name="Neutrophil",
            confidence=0.90,
            bbox_xyxy=(18.0, 2.0, 35.0, 18.0),
        ),
    ]
    summary, _ = build_summary(detections, CLASSES)
    return InferenceResult(
        sample_id="S001",
        created_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
        model=ModelInfo(name="blood-cell-yolo11", version="1.0.0", sha256="A" * 64),
        image=ImageInfo(filename="sample.png", width=40, height=20, sha256="B" * 64),
        detections=detections,
        summary=summary,
        qc=QCInfo(status="warning", warnings=["DETECTION_LIMIT_REACHED"]),
        runtime=TimingInfo(
            provider="CUDAExecutionProvider",
            preprocess_ms=1.0,
            inference_ms=2.0,
            postprocess_ms=1.0,
            total_ms=4.0,
        ),
    )


def test_write_result_json_round_trips_typed_result_and_warning(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "result.json"

    written = write_result_json(result(), destination)
    restored = InferenceResult.model_validate_json(destination.read_text("utf-8"))

    assert written == destination
    assert restored == result()
    assert "DETECTION_LIMIT_REACHED" in restored.qc.warnings


def test_write_detections_csv_has_fixed_header_and_one_row_per_detection(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "result.csv"

    write_detections_csv(result(), destination)

    with destination.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "sample_id",
        "model_name",
        "model_version",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    assert len(rows) == 2
    assert rows[0]["class_name"] == "RBC"


def test_write_annotated_image_preserves_dimensions_and_is_deterministic(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    write_annotated_image(result(), image_bytes(), first)
    write_annotated_image(result(), image_bytes(), second)

    with Image.open(first) as annotated:
        assert annotated.size == (40, 20)
        assert annotated.getpixel((1, 2)) != (255, 255, 255)
    assert first.read_bytes() == second.read_bytes()


def test_artifact_writers_leave_no_temporary_files(tmp_path: Path) -> None:
    write_result_json(result(), tmp_path / "result.json")
    write_detections_csv(result(), tmp_path / "result.csv")
    write_annotated_image(result(), image_bytes(), tmp_path / "annotated.png")

    assert list(tmp_path.glob("*.tmp")) == []
