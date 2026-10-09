from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Protocol

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from bloodsmear.domain import Detection, InferenceOptions
from bloodsmear.errors import (
    GPUUnavailableError,
    InferenceFailedError,
    ModelValidationError,
)
from bloodsmear.image_ops import letterbox
from bloodsmear.model_package import ModelPackage
from bloodsmear.postprocess import class_aware_nms, decode_yolo_output


class SessionNode(Protocol):
    name: str
    shape: Sequence[int | str | None]
    type: str


class SessionLike(Protocol):
    def get_inputs(self) -> Sequence[SessionNode]: ...

    def get_outputs(self) -> Sequence[SessionNode]: ...

    def get_providers(self) -> list[str]: ...

    def run(
        self,
        output_names: None,
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[NDArray[np.floating]]: ...


SessionFactory = Callable[[str, list[str]], SessionLike]


@dataclass(frozen=True)
class AdapterTimings:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


@dataclass(frozen=True)
class AdapterResult:
    detections: list[Detection]
    warnings: list[str]
    provider: str
    timings: AdapterTimings


class YOLO11OnnxAdapter:
    def __init__(
        self,
        package: ModelPackage,
        *,
        session_factory: SessionFactory | None = None,
        available_providers: Sequence[str] | None = None,
    ) -> None:
        self.package = package
        provider_names = list(available_providers or ort.get_available_providers())
        requested_providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in provider_names
        ]
        if not requested_providers:
            raise ModelValidationError("No supported ONNX Runtime provider is available")

        if session_factory is None:
            preload = getattr(ort, "preload_dlls", None)
            if preload is not None:
                preload(directory="")

            def session_factory(model_path: str, providers: list[str]) -> SessionLike:
                return ort.InferenceSession(model_path, providers=providers)

        self.requested_providers = requested_providers
        self.session = session_factory(str(package.model_path), requested_providers)
        self._validate_session()
        self.provider = self.session.get_providers()[0]

    def _validate_session(self) -> None:
        inputs = list(self.session.get_inputs())
        if len(inputs) != 1:
            raise ModelValidationError(
                f"Expected one model input, found {len(inputs)}"
            )
        model_input = inputs[0]
        if model_input.type != "tensor(float)":
            raise ModelValidationError(
                f"Expected FP32 model input, found {model_input.type}"
            )
        shape = list(model_input.shape)
        expected_size = self.package.manifest.input_size
        if (
            len(shape) != 4
            or shape[1] != 3
            or (isinstance(shape[2], int) and shape[2] != expected_size)
            or (isinstance(shape[3], int) and shape[3] != expected_size)
        ):
            raise ModelValidationError(
                f"Unexpected model input shape: {shape}; expected [1, 3, {expected_size}, {expected_size}]"
            )
        if not self.session.get_outputs():
            raise ModelValidationError("Model has no outputs")
        self.input_name = model_input.name
        self.input_shape = shape
        self.output_names = [node.name for node in self.session.get_outputs()]

    def infer(
        self,
        rgb: NDArray[np.uint8],
        options: InferenceOptions,
    ) -> AdapterResult:
        if options.require_gpu and self.provider != "CUDAExecutionProvider":
            raise GPUUnavailableError("CUDAExecutionProvider is required but unavailable")

        preprocess_started = perf_counter_ns()
        tensor, transform = letterbox(rgb, self.package.manifest.input_size)
        preprocess_ms = _elapsed_ms(preprocess_started)

        inference_started = perf_counter_ns()
        try:
            outputs = self.session.run(None, {self.input_name: tensor})
        except Exception as exc:
            raise InferenceFailedError("ONNX inference failed") from exc
        inference_ms = _elapsed_ms(inference_started)

        postprocess_started = perf_counter_ns()
        if not outputs:
            raise ModelValidationError("Model returned no outputs")
        try:
            decoded = decode_yolo_output(
                outputs[0],
                transform,
                self.package.manifest.classes,
                options.confidence_threshold,
            )
        except ValueError as exc:
            raise ModelValidationError("Model output layout is invalid") from exc
        detections, warnings = class_aware_nms(
            decoded,
            options.iou_threshold,
            options.max_detections,
        )
        if self.provider != "CUDAExecutionProvider":
            warnings.insert(0, "CPU_FALLBACK")
        postprocess_ms = _elapsed_ms(postprocess_started)

        return AdapterResult(
            detections=detections,
            warnings=warnings,
            provider=self.provider,
            timings=AdapterTimings(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
            ),
        )


def _elapsed_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000.0
