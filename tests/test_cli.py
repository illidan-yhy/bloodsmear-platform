from __future__ import annotations

import json
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
from bloodsmear.config import AppSettings
from bloodsmear.errors import ModelNotFoundError


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


def serve_settings(tmp_path: Path, require_gpu: bool) -> AppSettings:
    return AppSettings(
        require_gpu=require_gpu, port=18431, output_dir=tmp_path / "outputs",
        database_path=tmp_path / "data" / "jobs.db", jobs_root=tmp_path / "data" / "jobs",
        log_dir=tmp_path / "logs", log_to_console=False,
    )


def test_serve_required_gpu_stops_before_database_and_http_start(tmp_path, monkeypatch, capsys):
    settings = serve_settings(tmp_path, True)
    monkeypatch.setattr("bloodsmear.cli.AppSettings", lambda: settings)
    service = FakeService()
    service.adapter.provider = "CPUExecutionProvider"
    install_fake_service(monkeypatch, service)
    server_calls = []
    monkeypatch.setattr("bloodsmear.cli.run_http_server", lambda *args, **kwargs: server_calls.append(args))

    assert main(["serve"]) == 1
    assert server_calls == []
    assert not settings.database_path.exists()
    stderr = capsys.readouterr().err
    assert "GPU_UNAVAILABLE" in stderr
    assert "驱动" in stderr and "VC++" in stderr
    assert "Traceback" not in stderr
    state = json.loads((settings.log_dir / "startup-status.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["error_code"] == "GPU_UNAVAILABLE"
    assert state["provider"] == "CPUExecutionProvider"
    assert state["port"] == 18431 and state["require_gpu"] is True


@pytest.mark.parametrize(("require_gpu", "provider"), [
    (True, "CUDAExecutionProvider"), (False, "CPUExecutionProvider"),
])
def test_serve_allowed_provider_clears_old_gpu_failure(tmp_path, monkeypatch, require_gpu, provider):
    settings = serve_settings(tmp_path, require_gpu)
    settings.log_dir.mkdir(parents=True)
    record = settings.log_dir / "startup-status.json"
    record.write_text(json.dumps({"status": "failed", "error_code": "GPU_UNAVAILABLE", "port": 18431}), encoding="utf-8")
    monkeypatch.setattr("bloodsmear.cli.AppSettings", lambda: settings)
    service = FakeService()
    service.adapter.provider = provider
    install_fake_service(monkeypatch, service)
    states_seen_by_http_server = []
    def run_server(*args, **kwargs):
        states_seen_by_http_server.append(json.loads(record.read_text(encoding="utf-8")))
    monkeypatch.setattr("bloodsmear.cli.run_http_server", run_server)

    assert main(["serve"]) == 0
    assert len(states_seen_by_http_server) == 1
    assert states_seen_by_http_server[0]["status"] == "starting"
    assert states_seen_by_http_server[0]["error_code"] is None
    assert states_seen_by_http_server[0]["provider"] == provider
    state = json.loads(record.read_text(encoding="utf-8"))
    assert state["status"] == "stopped" and state["error_code"] is None


def test_serve_model_failure_replaces_previous_gpu_failure(tmp_path, monkeypatch):
    settings = serve_settings(tmp_path, True)
    settings.log_dir.mkdir(parents=True)
    record = settings.log_dir / "startup-status.json"
    record.write_text(json.dumps({"status": "failed", "error_code": "GPU_UNAVAILABLE", "port": 18431}), encoding="utf-8")
    monkeypatch.setattr("bloodsmear.cli.AppSettings", lambda: settings)
    def missing_model(_settings):
        raise ModelNotFoundError("missing")
    monkeypatch.setattr("bloodsmear.cli.InferenceService.from_settings", missing_model)

    assert main(["serve"]) == 1
    state = json.loads(record.read_text(encoding="utf-8"))
    assert state["status"] == "failed" and state["error_code"] == "MODEL_NOT_FOUND"
    assert state["provider"] is None
