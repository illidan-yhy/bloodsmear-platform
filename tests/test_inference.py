from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from bloodsmear.adapters.yolo11_onnx import AdapterResult, AdapterTimings
from bloodsmear.config import AppSettings
from bloodsmear.domain import Detection, InferenceOptions
from bloodsmear.inference import InferenceService
from bloodsmear.model_package import ModelManifest, ModelPackage


CLASSES = [
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelets",
    "RBC",
]


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (40, 20), color=(255, 0, 0)).save(stream, format="PNG")
    return stream.getvalue()


def package(tmp_path: Path) -> ModelPackage:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    manifest = ModelManifest(
        name="blood-cell-yolo11",
        version="1.0.0",
        model_file="model.onnx",
        sha256="A" * 64,
        input_size=640,
        input_dtype="float32",
        classes=CLASSES,
        confidence_threshold=0.25,
        iou_threshold=0.70,
        max_detections=3000,
        license_status="pending_publisher_confirmation",
    )
    return ModelPackage(
        root=tmp_path,
        model_path=model_path,
        manifest=manifest,
        labels={str(index): name for index, name in enumerate(CLASSES)},
    )


@dataclass
class FakeAdapter:
    calls: int = 0

    def infer(self, rgb: np.ndarray, options: InferenceOptions) -> AdapterResult:
        self.calls += 1
        assert rgb.shape == (20, 40, 3)
        assert options.confidence_threshold == 0.25
        return AdapterResult(
            detections=[
                Detection(
                    class_id=6,
                    class_name="RBC",
                    confidence=0.95,
                    bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
                ),
                Detection(
                    class_id=4,
                    class_name="Neutrophil",
                    confidence=0.90,
                    bbox_xyxy=(12.0, 2.0, 22.0, 12.0),
                ),
            ],
            warnings=["DETECTION_LIMIT_REACHED"],
            provider="CUDAExecutionProvider",
            timings=AdapterTimings(
                preprocess_ms=2.0,
                inference_ms=3.0,
                postprocess_ms=1.0,
            ),
        )


def service(tmp_path: Path, adapter: FakeAdapter) -> InferenceService:
    return InferenceService(
        package=package(tmp_path),
        adapter=adapter,
        settings=AppSettings(),
        clock=lambda: datetime(2026, 10, 3, tzinfo=timezone.utc),
        id_factory=lambda: "generated-id",
    )


def test_infer_bytes_builds_traceable_stable_result(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    subject = service(tmp_path, adapter)
    data = png_bytes()

    result = subject.infer_bytes(data, "../patient.png", sample_id="S001")

    assert result.sample_id == "S001"
    assert result.model.name == "blood-cell-yolo11"
    assert result.model.version == "1.0.0"
    assert result.model.sha256 == "A" * 64
    assert result.image.filename == "patient.png"
    assert result.image.sha256 == hashlib.sha256(data).hexdigest().upper()
    assert list(result.summary.cell_counts) == CLASSES
    assert result.summary.cell_counts["RBC"] == 1
    assert result.summary.cell_counts["Neutrophil"] == 1
    assert result.qc.status == "warning"
    assert result.qc.warnings == ["DETECTION_LIMIT_REACHED"]
    assert result.runtime.provider == "CUDAExecutionProvider"
    assert result.runtime.preprocess_ms == 2.0
    assert result.runtime.inference_ms == 3.0
    assert result.runtime.postprocess_ms == 1.0
    assert result.runtime.total_ms == 6.0


def test_infer_bytes_generates_sample_id_when_absent(tmp_path: Path) -> None:
    result = service(tmp_path, FakeAdapter()).infer_bytes(
        png_bytes(), "sample.png", sample_id=None
    )

    assert result.sample_id == "generated-id"


def test_infer_bytes_is_deterministic_for_deterministic_adapter(tmp_path: Path) -> None:
    subject = service(tmp_path, FakeAdapter())
    data = png_bytes()

    first = subject.infer_bytes(data, "sample.png", sample_id="same")
    second = subject.infer_bytes(data, "sample.png", sample_id="same")

    assert first.summary == second.summary
    assert first.detections == second.detections
    assert first.qc == second.qc
