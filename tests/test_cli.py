from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from bloodsmear.cli import main
from bloodsmear.domain import (
    Detection,
    ImageInfo,
    InferenceResult,
    ModelInfo,
    QCInfo,
    TimingInfo,
    build_summary,
)
from bloodsmear.errors import InferenceFailedError


CLASSES = [
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelets",
    "RBC",
]


def write_image(path: Path) -> None:
    stream = BytesIO()
    Image.new("RGB", (40, 20), color=(255, 255, 255)).save(stream, format="PNG")
    path.write_bytes(stream.getvalue())


def result() -> InferenceResult:
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
        sample_id="sample",
        created_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
        model=ModelInfo(name="blood-cell-yolo11", version="1.0.0", sha256="A" * 64),
        image=ImageInfo(filename="sample.png", width=40, height=20, sha256="B" * 64),
        detections=detections,
        summary=summary,
        qc=QCInfo(status="pass", warnings=[]),
        runtime=TimingInfo(
            provider="CUDAExecutionProvider",
            preprocess_ms=1.0,
            inference_ms=2.0,
            postprocess_ms=1.0,
            total_ms=4.0,
        ),
    )


class FakeService:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.adapter = SimpleNamespace(provider="CUDAExecutionProvider")
        self.package = SimpleNamespace(
            manifest=SimpleNamespace(
                name="blood-cell-yolo11",
                version="1.0.0",
                sha256="A" * 64,
                classes=CLASSES,
                license_status="pending_publisher_confirmation",
            )
        )

    def infer_bytes(self, *_args, **_kwargs) -> InferenceResult:
        if self.failure:
            raise self.failure
        return result()


def install_fake_service(monkeypatch, service: FakeService) -> None:
    monkeypatch.setattr(
        "bloodsmear.cli.InferenceService.from_settings",
        lambda _settings: service,
    )


def test_cli_infer_writes_three_artifacts_and_prints_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    image = tmp_path / "sample.png"
    output = tmp_path / "output"
    write_image(image)
    install_fake_service(monkeypatch, FakeService())

    exit_code = main(["infer", str(image), "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert (output / "result.json").is_file()
    assert (output / "detections.csv").is_file()
    assert (output / "annotated.png").is_file()
    assert str(output / "result.json") in captured.out
    assert str(output / "detections.csv") in captured.out
    assert str(output / "annotated.png") in captured.out


def test_cli_model_info_prints_model_traceability(
    monkeypatch,
    capsys,
) -> None:
    install_fake_service(monkeypatch, FakeService())

    exit_code = main(["model-info"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "blood-cell-yolo11" in captured.out
    assert "1.0.0" in captured.out
    assert "A" * 64 in captured.out
    assert "CUDAExecutionProvider" in captured.out


def test_cli_missing_input_returns_exit_code_2_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(["infer", str(tmp_path / "missing.png")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "does not exist" in captured.err
    assert "Traceback" not in captured.err


def test_cli_inference_failure_returns_exit_code_1_without_traceback(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    image = tmp_path / "sample.png"
    write_image(image)
    install_fake_service(monkeypatch, FakeService(failure=InferenceFailedError("failed")))

    exit_code = main(["infer", str(image)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "INFERENCE_FAILED: failed" in captured.err
    assert "Traceback" not in captured.err


def test_cli_debug_reraises_inference_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = tmp_path / "sample.png"
    write_image(image)
    install_fake_service(monkeypatch, FakeService(failure=InferenceFailedError("failed")))

    with pytest.raises(InferenceFailedError):
        main(["--debug", "infer", str(image)])
