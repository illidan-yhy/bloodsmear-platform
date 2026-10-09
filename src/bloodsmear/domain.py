from __future__ import annotations

from datetime import datetime
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


WBC_CLASSES = (
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
)


class SampleMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    patient_id: str | None = Field(default=None, max_length=128, repr=False)
    stain_method: str | None = Field(default=None, max_length=256, repr=False)
    scanner_model: str | None = Field(default=None, max_length=256, repr=False)
    magnification: str | None = Field(default=None, max_length=256, repr=False)
    pixel_resolution: str | None = Field(default=None, max_length=256, repr=False)
    operator: str | None = Field(default=None, max_length=256, repr=False)
    notes: str | None = Field(default=None, max_length=2000, repr=False)

    @field_validator("*", mode="before")
    @classmethod
    def normalize_optional_text(cls, value):
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class Detection(BaseModel):
    model_config = ConfigDict(frozen=True)

    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_box_order(self) -> "Detection":
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must have positive width and height")
        return self


class InferenceOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_detections: int = Field(default=3000, gt=0)
    require_gpu: bool = False


class TimingInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    preprocess_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    postprocess_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class QCInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["pass", "warning", "error"]
    warnings: list[str] = Field(default_factory=list)


class ResultSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_detected_cells: int = Field(ge=0)
    cell_counts: dict[str, int]
    all_cell_ratios: dict[str, float | None]
    wbc_differential_ratios: dict[str, float | None]


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class ImageInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class InferenceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    sample_id: str
    metadata: SampleMetadata | None = None
    created_at: datetime
    model: ModelInfo
    image: ImageInfo
    detections: list[Detection]
    summary: ResultSummary
    qc: QCInfo
    runtime: TimingInfo


def build_summary(
    detections: Sequence[Detection],
    classes: Sequence[str],
) -> tuple[ResultSummary, list[str]]:
    counts = {class_name: 0 for class_name in classes}
    for detection in detections:
        if (
            detection.class_id >= len(classes)
            or classes[detection.class_id] != detection.class_name
        ):
            raise ValueError(
                "Detection class mapping does not match the configured class order"
            )
        counts[detection.class_name] += 1

    total = sum(counts.values())
    all_ratios: dict[str, float | None]
    if total:
        all_ratios = {name: count / total for name, count in counts.items()}
    else:
        all_ratios = {name: None for name in classes}

    wbc_names = [name for name in classes if name in WBC_CLASSES]
    wbc_total = sum(counts[name] for name in wbc_names)
    warnings: list[str] = []
    if wbc_total:
        wbc_ratios = {name: counts[name] / wbc_total for name in wbc_names}
    else:
        wbc_ratios = {name: None for name in wbc_names}
        warnings.append("NO_WBC_DETECTED")

    return (
        ResultSummary(
            total_detected_cells=total,
            cell_counts=counts,
            all_cell_ratios=all_ratios,
            wbc_differential_ratios=wbc_ratios,
        ),
        warnings,
    )
