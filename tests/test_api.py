from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from bloodsmear.api import create_app
from bloodsmear.config import AppSettings
from bloodsmear.domain import (
    Detection,
    ImageInfo,
    InferenceResult,
    ModelInfo,
    QCInfo,
    TimingInfo,
    build_summary,
)
from bloodsmear.errors import InvalidImageError


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
    Image.new("RGB", (40, 20), color=(255, 255, 255)).save(stream, format="PNG")
    return stream.getvalue()


def inference_result() -> InferenceResult:
    detections = [
        Detection(
            class_id=6,
            class_name="RBC",
            confidence=0.95,
            bbox_xyxy=(1.0, 2.0, 15.0, 18.0),
        )
    ]
    summary, _ = build_summary(detections, CLASSES)
    return InferenceResult(
        sample_id="generated-id",
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


class FakeService:
    def __init__(self, *, provider: str = "CUDAExecutionProvider") -> None:
        self.adapter = SimpleNamespace(provider=provider)
        self.package = SimpleNamespace(
            manifest=SimpleNamespace(
                name="blood-cell-yolo11",
                version="1.0.0",
                sha256="A" * 64,
                classes=CLASSES,
                license_status="pending_publisher_confirmation",
            )
        )

    def infer_bytes(
        self,
        data: bytes,
        filename: str,
        sample_id: str | None,
        options=None,
        metadata=None,
    ) -> InferenceResult:
        if data == b"invalid":
            raise InvalidImageError("Unable to decode image data")
        result = inference_result()
        return result.model_copy(update={"metadata": metadata})


def client(
    tmp_path: Path,
    *,
    require_gpu: bool = False,
    provider: str = "CUDAExecutionProvider",
    report_service=None,
) -> TestClient:
    settings = AppSettings(output_dir=tmp_path, require_gpu=require_gpu)
    return TestClient(create_app(FakeService(provider=provider), settings, report_service=report_service))


def test_infer_endpoint_returns_result_and_artifact_urls(tmp_path: Path) -> None:
    response = client(tmp_path).post(
        "/api/v1/infer",
        files={"image": ("sample.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["detections"][0]["class_name"] == "RBC"
    assert payload["summary"]["cell_counts"]["RBC"] == 1
    assert payload["runtime"]["provider"] == "CUDAExecutionProvider"
    assert set(payload["artifacts"]) == {"json", "csv", "annotated_image", "excel", "pdf"}
    assert payload["artifacts"]["excel"].endswith("/report.xlsx")
    assert payload["artifacts"]["pdf"].endswith("/report.pdf")


def test_invalid_image_returns_stable_400_error(tmp_path: Path) -> None:
    response = client(tmp_path).post(
        "/api/v1/infer",
        files={"image": ("sample.png", b"invalid", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_report_failure_preserves_result_and_adds_warning(tmp_path: Path) -> None:
    class FailingReports:
        def generate_single(self, *_args, **_kwargs):
            raise RuntimeError("private path")
    response = client(tmp_path, report_service=FailingReports()).post(
        "/api/v1/infer", files={"image": ("sample.png", png_bytes(), "image/png")}
    )
    payload = response.json()
    assert response.status_code == 200
    assert "REPORT_GENERATION_FAILED" in payload["qc"]["warnings"]
    assert payload["artifacts"]["excel"] is None
    assert payload["artifacts"]["json"]


def test_single_infer_accepts_optional_metadata(tmp_path: Path) -> None:
    response = client(tmp_path).post(
        "/api/v1/infer",
        data={"sample_id": "S001", "patient_id": "P001", "stain_method": "瑞氏染色"},
        files={"image": ("sample.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["metadata"]["patient_id"] == "P001"
    assert response.json()["metadata"]["stain_method"] == "瑞氏染色"


def test_current_model_endpoint_returns_traceability(tmp_path: Path) -> None:
    response = client(tmp_path).get("/api/v1/models/current")

    assert response.status_code == 200
    assert response.json() == {
        "name": "blood-cell-yolo11",
        "version": "1.0.0",
        "sha256": "A" * 64,
        "classes": CLASSES,
        "license_status": "pending_publisher_confirmation",
        "provider": "CUDAExecutionProvider",
    }


def test_liveness_stays_up_when_required_gpu_is_unavailable(tmp_path: Path) -> None:
    subject = client(tmp_path, require_gpu=True, provider="CPUExecutionProvider")

    assert subject.get("/health/live").status_code == 200
    readiness = subject.get("/health/ready")
    assert readiness.status_code == 503
    assert readiness.json()["error"]["code"] == "GPU_UNAVAILABLE"


def test_home_and_htmx_result_render_complete_workflow(tmp_path: Path) -> None:
    subject = client(tmp_path)

    home = subject.get("/")
    result = subject.post(
        "/ui/infer",
        headers={"HX-Request": "true"},
        files={"image": ("sample.png", png_bytes(), "image/png")},
    )

    assert home.status_code == 200
    assert "外周血涂片推理" in home.text
    assert "https://unpkg.com" not in home.text
    assert "/static/htmx.min.js" in home.text
    assert result.status_code == 200
    assert "RBC" in result.text
    assert "DETECTION_LIMIT_REACHED" in result.text
    assert "下载标注图" in result.text
    assert "/artifacts/" in result.text


def test_request_logging_uses_request_id_without_filename_or_body(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="bloodsmear.api")

    response = client(tmp_path).post(
        "/api/v1/infer",
        files={"image": ("secret-patient.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert "request_complete" in caplog.text
    assert "secret-patient.png" not in caplog.text
    assert "multipart/form-data" not in caplog.text
