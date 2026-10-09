from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from bloodsmear.domain import SampleMetadata


class BatchMode(StrEnum):
    INDEPENDENT = "independent"
    GROUPED = "grouped"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"


class ItemStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_JOB_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.PARTIAL_FAILED,
    JobStatus.FAILED,
}


class BatchJob(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    mode: BatchMode
    sample_id: str | None = None
    metadata: SampleMetadata | None = None
    status: JobStatus = JobStatus.QUEUED
    total_items: int = Field(ge=1)
    completed_items: int = Field(default=0, ge=0)
    failed_items: int = Field(default=0, ge=0)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_expires_at: datetime
    result_expires_at: datetime
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_grouped_sample(self) -> "BatchJob":
        normalized = self.sample_id.strip() if self.sample_id else None
        if self.mode == BatchMode.GROUPED and not normalized:
            raise ValueError("sample_id is required for grouped jobs")
        if self.mode == BatchMode.INDEPENDENT and self.metadata and self.metadata.patient_id:
            raise ValueError("patient_id is not allowed for independent jobs")
        if normalized != self.sample_id:
            object.__setattr__(self, "sample_id", normalized)
        if self.completed_items + self.failed_items > self.total_items:
            raise ValueError("job progress exceeds total_items")
        return self


class BatchItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    original_filename: str = Field(min_length=1)
    stored_filename: str = Field(min_length=1)
    input_path: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    view_id: str = Field(min_length=1)
    status: ItemStatus = ItemStatus.PENDING
    result_directory: str | None = None
    result_summary_json: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    inference_ms: float | None = Field(default=None, ge=0.0)

    @field_validator("input_path", "result_directory")
    @classmethod
    def paths_must_be_relative(cls, value: str | None) -> str | None:
        if value is not None and Path(value).is_absolute():
            raise ValueError("stored paths must be relative")
        if value is not None and ".." in Path(value).parts:
            raise ValueError("stored paths must be relative")
        return value


class BatchJobDetail(BatchJob):
    items: list[BatchItem] = Field(default_factory=list)
