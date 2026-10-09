from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bloodsmear.batch.aggregation import BatchSummary
from bloodsmear.batch.domain import BatchJobDetail
from bloodsmear.domain import InferenceResult
from bloodsmear.errors import ReportGenerationError


SHANGHAI = ZoneInfo("Asia/Shanghai")
FORMULA_PREFIXES = ("=", "+", "-", "@")


def shanghai_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(SHANGHAI)


def excel_safe_text(value: str | None) -> str:
    if value is None:
        return "—"
    if value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


class SingleReportContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    result: InferenceResult
    license_status: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def finite_ratios(self) -> "SingleReportContext":
        _validate_finite_ratios(self.result)
        return self


class BatchReportContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    job: BatchJobDetail
    summary: BatchSummary
    results: list[InferenceResult]
    license_status: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_batch(self) -> "BatchReportContext":
        hashes = {result.model.sha256 for result in self.results}
        if len(hashes) > 1:
            raise ReportGenerationError("Batch contains mixed model hashes")
        for result in self.results:
            _validate_finite_ratios(result)
        return self


class ReportContextFactory:
    @staticmethod
    def single(result: InferenceResult, license_status: str) -> SingleReportContext:
        return SingleReportContext(result=result, license_status=license_status)

    @staticmethod
    def batch(
        job: BatchJobDetail,
        summary: BatchSummary,
        results: list[InferenceResult],
        license_status: str,
    ) -> BatchReportContext:
        ordered_results = sorted(results, key=lambda result: result.image.filename)
        return BatchReportContext(
            job=job,
            summary=summary,
            results=ordered_results,
            license_status=license_status,
        )


def _validate_finite_ratios(result: InferenceResult) -> None:
    mappings = (
        result.summary.all_cell_ratios,
        result.summary.wbc_differential_ratios,
    )
    for mapping in mappings:
        for value in mapping.values():
            if value is not None and not math.isfinite(value):
                raise ReportGenerationError("Report ratios must be finite")
