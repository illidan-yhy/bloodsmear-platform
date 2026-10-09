from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from bloodsmear.adapters.yolo11_onnx import AdapterResult, YOLO11OnnxAdapter
from bloodsmear.config import AppSettings
from bloodsmear.domain import (
    ImageInfo,
    InferenceOptions,
    InferenceResult,
    ModelInfo,
    QCInfo,
    SampleMetadata,
    TimingInfo,
    build_summary,
)
from bloodsmear.image_ops import decode_image
from bloodsmear.model_package import ModelPackage, load_model_package


class InferenceAdapter(Protocol):
    def infer(self, rgb, options: InferenceOptions) -> AdapterResult: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_sample_id() -> str:
    return uuid4().hex


class InferenceService:
    def __init__(
        self,
        *,
        package: ModelPackage,
        adapter: InferenceAdapter,
        settings: AppSettings,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[], str] = _new_sample_id,
    ) -> None:
        self.package = package
        self.adapter = adapter
        self.settings = settings
        self.clock = clock
        self.id_factory = id_factory

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "InferenceService":
        package = load_model_package(settings.model_dir)
        adapter = YOLO11OnnxAdapter(package)
        return cls(package=package, adapter=adapter, settings=settings)

    def infer_bytes(
        self,
        data: bytes,
        filename: str,
        sample_id: str | None,
        options: InferenceOptions | None = None,
        metadata: SampleMetadata | None = None,
    ) -> InferenceResult:
        decoded = decode_image(data, filename, self.settings.max_upload_bytes)
        active_options = options or InferenceOptions(
            confidence_threshold=self.settings.confidence_threshold,
            iou_threshold=self.settings.iou_threshold,
            max_detections=self.settings.max_detections,
            require_gpu=self.settings.require_gpu,
        )
        adapter_result = self.adapter.infer(decoded.rgb, active_options)
        summary, summary_warnings = build_summary(
            adapter_result.detections,
            self.package.manifest.classes,
        )
        warnings = list(
            dict.fromkeys([*adapter_result.warnings, *summary_warnings])
        )
        timings = adapter_result.timings
        total_ms = (
            timings.preprocess_ms + timings.inference_ms + timings.postprocess_ms
        )

        return InferenceResult(
            sample_id=sample_id or self.id_factory(),
            metadata=metadata,
            created_at=self.clock(),
            model=ModelInfo(
                name=self.package.manifest.name,
                version=self.package.manifest.version,
                sha256=self.package.manifest.sha256.upper(),
            ),
            image=ImageInfo(
                filename=decoded.filename,
                width=decoded.width,
                height=decoded.height,
                sha256=decoded.sha256,
            ),
            detections=adapter_result.detections,
            summary=summary,
            qc=QCInfo(
                status="warning" if warnings else "pass",
                warnings=warnings,
            ),
            runtime=TimingInfo(
                provider=adapter_result.provider,
                preprocess_ms=timings.preprocess_ms,
                inference_ms=timings.inference_ms,
                postprocess_ms=timings.postprocess_ms,
                total_ms=total_ms,
            ),
        )
