from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bloodsmear.batch.domain import (
    BatchItem,
    BatchJob,
    BatchMode,
    ItemStatus,
    JobStatus,
)
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.errors import JobNotFoundError


NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def make_job(job_id: str = "job-1", total: int = 2) -> BatchJob:
    return BatchJob(
        id=job_id,
        mode=BatchMode.INDEPENDENT,
        status=JobStatus.QUEUED,
        total_items=total,
        created_at=NOW,
        input_expires_at=NOW + timedelta(hours=24),
        result_expires_at=NOW + timedelta(days=7),
    )


def make_items(job_id: str = "job-1", total: int = 2) -> list[BatchItem]:
    return [
        BatchItem(
            id=f"item-{index}",
            job_id=job_id,
            ordinal=index,
            original_filename=f"field-{index}.png",
            stored_filename=f"item-{index}.png",
            input_path=f"inputs/item-{index}.png",
            sample_id=f"sample-{index}",
            view_id=f"view-{index}",
            status=ItemStatus.PENDING,
        )
        for index in range(total)
    ]


def repository(tmp_path: Path) -> BatchRepository:
    repo = BatchRepository(tmp_path / "batch.db")
    repo.initialize()
    return repo


def test_initialize_is_idempotent_and_create_job_is_atomic(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.initialize()
    promoted: list[bool] = []

    repo.create_job(make_job(), make_items(), lambda: promoted.append(True))
    detail = repo.get_job("job-1")

    assert promoted == [True]
    assert detail.status == JobStatus.QUEUED
    assert [item.ordinal for item in detail.items] == [0, 1]


def test_create_job_rolls_back_when_promotion_fails(tmp_path: Path) -> None:
    repo = repository(tmp_path)

    with pytest.raises(RuntimeError, match="promotion failed"):
        repo.create_job(
            make_job(),
            make_items(),
            lambda: (_ for _ in ()).throw(RuntimeError("promotion failed")),
        )

    with pytest.raises(JobNotFoundError):
        repo.get_job("job-1")


def test_claim_next_job_is_exactly_once_across_connections(tmp_path: Path) -> None:
    database = tmp_path / "batch.db"
    first = BatchRepository(database)
    second = BatchRepository(database)
    first.initialize()
    first.create_job(make_job(), make_items(), lambda: None)

    claimed = first.claim_next_job(NOW + timedelta(minutes=1))
    duplicate = second.claim_next_job(NOW + timedelta(minutes=1))

    assert claimed is not None
    assert claimed.id == "job-1"
    assert claimed.status == JobStatus.RUNNING
    assert duplicate is None


def test_item_claim_reports_actual_database_transition(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.create_job(make_job(), make_items(), lambda: None)
    repo.claim_next_job(NOW)
    assert repo.mark_item_running("item-0") is True
    assert repo.mark_item_running("item-0") is False
    assert repo.mark_item_running("missing") is False


def test_job_claim_returns_none_when_update_did_not_succeed(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.create_job(make_job(), make_items(), lambda: None)
    with repo._connect() as connection:
        connection.execute("CREATE TRIGGER deny_claim BEFORE UPDATE OF status ON batch_jobs WHEN NEW.status = 'running' BEGIN SELECT RAISE(IGNORE); END")
    assert repo.claim_next_job(NOW) is None
    assert repo.get_job("job-1").status == JobStatus.QUEUED


def test_item_updates_progress_and_partial_finalization(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.create_job(make_job(), make_items(), lambda: None)
    repo.claim_next_job(NOW)

    repo.mark_item_running("item-0")
    repo.complete_item("item-0", "items/item-0", '{"ok": true}', 12.5)
    repo.mark_item_running("item-1")
    repo.fail_item("item-1", "ITEM_INFERENCE_FAILED", "failed")
    repo.finalize_job("job-1", JobStatus.PARTIAL_FAILED, NOW + timedelta(minutes=2))

    detail = repo.get_job("job-1")
    assert detail.status == JobStatus.PARTIAL_FAILED
    assert detail.completed_items == 1
    assert detail.failed_items == 1
    assert detail.items[0].status == ItemStatus.COMPLETED
    assert detail.items[1].status == ItemStatus.FAILED


@pytest.mark.parametrize(
    ("item_status", "job_status"),
    [
        (ItemStatus.COMPLETED, JobStatus.COMPLETED),
        (ItemStatus.FAILED, JobStatus.FAILED),
    ],
)
def test_all_success_or_failure_finalization(
    tmp_path: Path,
    item_status: ItemStatus,
    job_status: JobStatus,
) -> None:
    repo = repository(tmp_path)
    repo.create_job(make_job(total=1), make_items(total=1), lambda: None)
    repo.claim_next_job(NOW)
    repo.mark_item_running("item-0")
    if item_status == ItemStatus.COMPLETED:
        repo.complete_item("item-0", "items/item-0", "{}", 1.0)
    else:
        repo.fail_item("item-0", "ITEM_INFERENCE_FAILED", "failed")

    repo.finalize_job("job-1", job_status, NOW)

    assert repo.get_job("job-1").status == job_status


def test_recover_interrupted_preserves_completed_items(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.create_job(make_job(), make_items(), lambda: None)
    repo.claim_next_job(NOW)
    repo.mark_item_running("item-0")
    repo.complete_item("item-0", "items/item-0", "{}", 1.0)
    repo.mark_item_running("item-1")

    recovered = repo.recover_interrupted()
    detail = repo.get_job("job-1")

    assert recovered == 1
    assert detail.status == JobStatus.QUEUED
    assert detail.items[0].status == ItemStatus.COMPLETED
    assert detail.items[1].status == ItemStatus.PENDING
    assert detail.completed_items == 1


def test_delete_job_cascades_items(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.create_job(make_job(), make_items(), lambda: None)

    repo.delete_job("job-1")

    with pytest.raises(JobNotFoundError):
        repo.get_job("job-1")
    assert repo.count_items("job-1") == 0
