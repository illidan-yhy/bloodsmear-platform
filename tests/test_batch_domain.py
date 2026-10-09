from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from bloodsmear.batch.domain import (
    BatchItem,
    BatchJob,
    BatchMode,
    ItemStatus,
    JobStatus,
)
from bloodsmear.config import AppSettings
from bloodsmear.errors import (
    BatchTooManyFilesError,
    BatchTotalSizeExceededError,
    GroupSampleIdRequiredError,
    ItemInferenceFailedError,
    JobNotFoundError,
    JobNotReadyError,
    UnsafeJobPathError,
)


NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def make_job(mode: BatchMode, sample_id: str | None) -> BatchJob:
    return BatchJob(
        id="a" * 32,
        mode=mode,
        sample_id=sample_id,
        status=JobStatus.QUEUED,
        total_items=1,
        completed_items=0,
        failed_items=0,
        created_at=NOW,
        input_expires_at=NOW + timedelta(hours=24),
        result_expires_at=NOW + timedelta(days=7),
    )


def test_batch_enums_match_persisted_values() -> None:
    assert [value.value for value in BatchMode] == ["independent", "grouped"]
    assert [value.value for value in JobStatus] == [
        "queued",
        "running",
        "completed",
        "partial_failed",
        "failed",
    ]
    assert [value.value for value in ItemStatus] == [
        "pending",
        "running",
        "completed",
        "failed",
    ]


@pytest.mark.parametrize("sample_id", [None, "", "   "])
def test_grouped_job_requires_non_empty_sample_id(sample_id: str | None) -> None:
    with pytest.raises(ValidationError, match="sample_id"):
        make_job(BatchMode.GROUPED, sample_id)


def test_independent_job_accepts_null_sample_id() -> None:
    job = make_job(BatchMode.INDEPENDENT, None)
    assert job.sample_id is None


def test_batch_item_uses_relative_paths_only() -> None:
    with pytest.raises(ValidationError, match="relative"):
        BatchItem(
            id="b" * 32,
            job_id="a" * 32,
            ordinal=0,
            original_filename="sample.png",
            stored_filename="b.png",
            input_path="C:/outside/sample.png",
            sample_id="sample-b",
            view_id="view-1",
            status=ItemStatus.PENDING,
        )


def test_settings_use_batch_defaults() -> None:
    settings = AppSettings()
    assert settings.database_path == Path("data/bloodsmear.db")
    assert settings.jobs_root == Path("data/jobs")
    assert settings.batch_max_files == 100
    assert settings.batch_max_file_bytes == 25 * 1024 * 1024
    assert settings.batch_max_total_bytes == 500 * 1024 * 1024
    assert settings.input_retention_hours == 24
    assert settings.result_retention_days == 7
    assert settings.cleanup_interval_seconds == 3600


def test_settings_accept_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("BLOODSMEAR_BATCH_MAX_FILES", "12")
    monkeypatch.setenv("BLOODSMEAR_INPUT_RETENTION_HOURS", "6")
    monkeypatch.setenv("BLOODSMEAR_JOBS_ROOT", "custom/jobs")

    settings = AppSettings()

    assert settings.batch_max_files == 12
    assert settings.input_retention_hours == 6
    assert settings.jobs_root == Path("custom/jobs")


def test_batch_error_codes_are_stable() -> None:
    assert BatchTooManyFilesError.code == "BATCH_TOO_MANY_FILES"
    assert BatchTotalSizeExceededError.code == "BATCH_TOTAL_SIZE_EXCEEDED"
    assert GroupSampleIdRequiredError.code == "GROUP_SAMPLE_ID_REQUIRED"
    assert JobNotFoundError.code == "JOB_NOT_FOUND"
    assert JobNotReadyError.code == "JOB_NOT_READY"
    assert ItemInferenceFailedError.code == "ITEM_INFERENCE_FAILED"
    assert UnsafeJobPathError.code == "UNSAFE_JOB_PATH"
