from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from bloodsmear.domain import InferenceOptions
from bloodsmear.errors import (
    GPUUnavailableError,
    InferenceFailedError,
    ModelValidationError,
)
from bloodsmear.model_package import ModelManifest, ModelPackage
from bloodsmear.adapters.yolo11_onnx import YOLO11OnnxAdapter


CLASSES = [
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelets",
    "RBC",
]


@dataclass
class FakeNode:
    name: str
    shape: list[int]
    type: str = "tensor(float)"


class FakeSession:
    def __init__(
        self,
        output: np.ndarray,
        providers: list[str],
        *,
        input_shape: list[int] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.output = output
        self.providers = providers
        self.failure = failure
        self.input_shape = input_shape or [1, 3, 640, 640]
        self.last_feed: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[FakeNode]:
        return [FakeNode("images", self.input_shape)]

    def get_outputs(self) -> list[FakeNode]:
        return [FakeNode("output0", [1, 11, 1])]

    def get_providers(self) -> list[str]:
        return self.providers

    def run(
        self,
        output_names: None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        self.last_feed = input_feed
        if self.failure:
            raise self.failure
        return [self.output]


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


def output_for_layout(layout: str) -> np.ndarray:
    row = np.zeros((1, 11), dtype=np.float32)
    row[0, :4] = (320, 320, 100, 100)
    row[0, 10] = 0.9
    if layout == "channels-first":
        return row.T[None, :, :]
    return row[None, :, :]


@pytest.mark.parametrize("layout", ["channels-first", "channels-last"])
def test_adapter_requests_cuda_then_cpu_and_accepts_output_layouts(
    tmp_path: Path,
    layout: str,
) -> None:
    captured: dict[str, object] = {}
    session = FakeSession(
        output_for_layout(layout),
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    def factory(model_path: str, providers: list[str]) -> FakeSession:
        captured["model_path"] = model_path
        captured["providers"] = providers
        return session

    adapter = YOLO11OnnxAdapter(
        package(tmp_path),
        session_factory=factory,
        available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    result = adapter.infer(
        np.zeros((80, 100, 3), dtype=np.uint8),
        InferenceOptions(),
    )

    assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert session.last_feed is not None
    assert session.last_feed["images"].shape == (1, 3, 640, 640)
    assert session.last_feed["images"].dtype == np.float32
    assert result.provider == "CUDAExecutionProvider"
    assert result.detections[0].class_name == "RBC"


def test_adapter_rejects_cpu_fallback_when_gpu_is_required(tmp_path: Path) -> None:
    session = FakeSession(output_for_layout("channels-first"), ["CPUExecutionProvider"])
    adapter = YOLO11OnnxAdapter(
        package(tmp_path),
        session_factory=lambda *_args: session,
        available_providers=["CPUExecutionProvider"],
    )

    with pytest.raises(GPUUnavailableError):
        adapter.infer(
            np.zeros((80, 100, 3), dtype=np.uint8),
            InferenceOptions(require_gpu=True),
        )


def test_adapter_reports_cpu_fallback_in_optional_mode(tmp_path: Path) -> None:
    session = FakeSession(output_for_layout("channels-last"), ["CPUExecutionProvider"])
    adapter = YOLO11OnnxAdapter(
        package(tmp_path),
        session_factory=lambda *_args: session,
        available_providers=["CPUExecutionProvider"],
    )

    result = adapter.infer(
        np.zeros((80, 100, 3), dtype=np.uint8),
        InferenceOptions(require_gpu=False),
    )

    assert result.provider == "CPUExecutionProvider"
    assert "CPU_FALLBACK" in result.warnings


def test_adapter_rejects_invalid_model_input_shape(tmp_path: Path) -> None:
    session = FakeSession(
        output_for_layout("channels-first"),
        ["CPUExecutionProvider"],
        input_shape=[1, 1, 640, 640],
    )

    with pytest.raises(ModelValidationError, match="input shape"):
        YOLO11OnnxAdapter(
            package(tmp_path),
            session_factory=lambda *_args: session,
            available_providers=["CPUExecutionProvider"],
        )


def test_adapter_sanitizes_runtime_failure(tmp_path: Path) -> None:
    session = FakeSession(
        output_for_layout("channels-first"),
        ["CPUExecutionProvider"],
        failure=RuntimeError("SECRETPIXELS and model bytes"),
    )
    adapter = YOLO11OnnxAdapter(
        package(tmp_path),
        session_factory=lambda *_args: session,
        available_providers=["CPUExecutionProvider"],
    )

    with pytest.raises(InferenceFailedError) as captured:
        adapter.infer(
            np.zeros((80, 100, 3), dtype=np.uint8),
            InferenceOptions(),
        )

    assert str(captured.value) == "ONNX inference failed"
    assert "SECRETPIXELS" not in str(captured.value)
