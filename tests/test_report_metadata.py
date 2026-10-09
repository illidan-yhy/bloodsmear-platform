from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from bloodsmear.batch.domain import BatchItem, BatchJob, BatchMode, ItemStatus, JobStatus
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.domain import (
    ImageInfo,
    InferenceResult,
    ModelInfo,
    QCInfo,
    ResultSummary,
    SampleMetadata,
    TimingInfo,
)
from bloodsmear.errors import ReportGenerationError


NOW = datetime(2026, 10, 4, tzinfo=timezone.utc)


def test_metadata_normalizes_blank_values_and_hides_sensitive_repr() -> None:
    metadata = SampleMetadata(patient_id=" P001 ", notes=" secret note ", operator="   ")
    assert metadata.patient_id == "P001"
    assert metadata.notes == "secret note"
    assert metadata.operator is None
    assert "P001" not in repr(metadata)
    assert "secret note" not in repr(metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [("patient_id", "x" * 129), ("scanner_model", "x" * 257), ("notes", "x" * 2001)],
)
def test_metadata_enforces_length_limits(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        SampleMetadata(**{field: value})


def test_old_inference_json_without_metadata_remains_valid() -> None:
    payload = {
        "sample_id": "S001",
        "created_at": NOW.isoformat(),
        "model": {"name": "model", "version": "1", "sha256": "A" * 64},
        "image": {"filename": "x.png", "width": 10, "height": 10, "sha256": "B" * 64},
        "detections": [],
        "summary": ResultSummary(
            total_detected_cells=0,
            cell_counts={"RBC": 0},
            all_cell_ratios={"RBC": None},
            wbc_differential_ratios={},
        ).model_dump(),
        "qc": {"status": "pass", "warnings": []},
        "runtime": {
            "provider": "CPUExecutionProvider",
            "preprocess_ms": 0,
            "inference_ms": 0,
            "postprocess_ms": 0,
            "total_ms": 0,
        },
    }
    assert InferenceResult.model_validate(payload).metadata is None


def make_job(mode: BatchMode, metadata: SampleMetadata | None) -> BatchJob:
    return BatchJob(
        id="job",
        mode=mode,
        sample_id="S001" if mode == BatchMode.GROUPED else None,
        metadata=metadata,
        status=JobStatus.QUEUED,
        total_items=1,
        created_at=NOW,
        input_expires_at=NOW + timedelta(hours=24),
        result_expires_at=NOW + timedelta(days=7),
    )


def test_independent_batch_rejects_patient_id() -> None:
    with pytest.raises(ValidationError, match="patient_id"):
        make_job(BatchMode.INDEPENDENT, SampleMetadata(patient_id="P001"))


def test_batch_metadata_round_trips_sqlite(tmp_path: Path) -> None:
    repo = BatchRepository(tmp_path / "batch.db")
    repo.initialize()
    job = make_job(BatchMode.GROUPED, SampleMetadata(patient_id="P001", stain_method="Wright"))
    item = BatchItem(
        id="item", job_id="job", ordinal=0, original_filename="x.png",
        stored_filename="x.png", input_path="inputs/x.png", sample_id="S001",
        view_id="v1", status=ItemStatus.PENDING,
    )
    repo.create_job(job, [item], lambda: None)
    restored = repo.get_job("job")
    assert restored.metadata == job.metadata


def test_report_error_code_is_stable() -> None:
    assert ReportGenerationError.code == "REPORT_GENERATION_FAILED"
