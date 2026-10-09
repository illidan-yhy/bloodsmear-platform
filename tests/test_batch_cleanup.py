from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bloodsmear.batch.cleanup import BatchCleanup
from bloodsmear.batch.domain import (
    BatchItem,
    BatchJob,
    BatchMode,
    ItemStatus,
    JobStatus,
)
from bloodsmear.batch.manager import BatchManager
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage
from bloodsmear.config import AppSettings
from bloodsmear.errors import JobNotFoundError, JobNotReadyError, UnsafeJobPathError


NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def setup(tmp_path: Path):
    settings = AppSettings(
        database_path=tmp_path / "batch.db",
        jobs_root=tmp_path / "jobs",
        cleanup_interval_seconds=1,
    )
    repo = BatchRepository(settings.database_path)
    repo.initialize()
    storage = BatchStorage(settings)
    manager = BatchManager(repo, storage, settings)
    cleanup = BatchCleanup(repo, storage, interval_seconds=0.01)
    return repo, storage, manager, cleanup


def add_job(
    repo: BatchRepository,
    storage: BatchStorage,
    job_id: str,
    status: JobStatus,
    *,
    input_expired: bool,
    result_expired: bool,
) -> None:
    root = storage.job_root(job_id)
    (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "image.png").write_bytes(b"image")
    (root / "summary.json").write_text("{}", encoding="utf-8")
    job = BatchJob(
        id=job_id,
        mode=BatchMode.INDEPENDENT,
        status=status,
        total_items=1,
        completed_items=1 if status == JobStatus.COMPLETED else 0,
        created_at=NOW - timedelta(days=10),
        input_expires_at=NOW - timedelta(hours=1) if input_expired else NOW + timedelta(hours=1),
        result_expires_at=NOW - timedelta(hours=1) if result_expired else NOW + timedelta(days=1),
    )
    item = BatchItem(
        id=f"{job_id}-item",
        job_id=job_id,
        ordinal=0,
        original_filename="image.png",
        stored_filename="image.png",
        input_path="inputs/image.png",
        sample_id="sample",
        view_id="view",
        status=ItemStatus.COMPLETED if status == JobStatus.COMPLETED else ItemStatus.PENDING,
    )
    repo.create_job(job, [item], lambda: None)


def test_input_expiry_removes_only_inputs_for_terminal_job(tmp_path: Path) -> None:
    repo, storage, _, cleanup = setup(tmp_path)
    add_job(repo, storage, "job", JobStatus.COMPLETED, input_expired=True, result_expired=False)

    report = cleanup.run_once(NOW)

    assert report.inputs_removed == 1
    assert report.jobs_removed == 0
    assert not (storage.job_root("job") / "inputs").exists()
    assert (storage.job_root("job") / "summary.json").is_file()
    assert repo.get_job("job").id == "job"


def test_result_expiry_removes_exact_job_and_database_rows(tmp_path: Path) -> None:
    repo, storage, _, cleanup = setup(tmp_path)
    add_job(repo, storage, "expired", JobStatus.COMPLETED, input_expired=True, result_expired=True)
    add_job(repo, storage, "sibling", JobStatus.COMPLETED, input_expired=False, result_expired=False)

    report = cleanup.run_once(NOW)

    assert report.jobs_removed == 1
    assert not storage.job_root("expired").exists()
    with pytest.raises(JobNotFoundError):
        repo.get_job("expired")
    assert storage.job_root("sibling").exists()
    assert repo.get_job("sibling").id == "sibling"


def test_cleanup_does_not_remove_active_job(tmp_path: Path) -> None:
    repo, storage, _, cleanup = setup(tmp_path)
    add_job(repo, storage, "active", JobStatus.RUNNING, input_expired=True, result_expired=True)

    report = cleanup.run_once(NOW)

    assert report.inputs_removed == 0
    assert report.jobs_removed == 0
    assert storage.job_root("active").exists()


def test_manual_delete_requires_terminal_job_and_reports_unknown(tmp_path: Path) -> None:
    repo, storage, manager, _ = setup(tmp_path)
    add_job(repo, storage, "active", JobStatus.QUEUED, input_expired=False, result_expired=False)

    with pytest.raises(JobNotReadyError):
        manager.delete_job("active")
    with pytest.raises(JobNotFoundError):
        manager.delete_job("missing")


def test_manual_delete_removes_only_target_terminal_job(tmp_path: Path) -> None:
    repo, storage, manager, _ = setup(tmp_path)
    add_job(repo, storage, "target", JobStatus.COMPLETED, input_expired=False, result_expired=False)
    add_job(repo, storage, "sibling", JobStatus.COMPLETED, input_expired=False, result_expired=False)

    manager.delete_job("target")

    assert not storage.job_root("target").exists()
    assert storage.job_root("sibling").exists()
    assert repo.get_job("sibling").id == "sibling"


def test_guarded_remove_rejects_traversal(tmp_path: Path) -> None:
    _, storage, _, _ = setup(tmp_path)

    with pytest.raises(UnsafeJobPathError):
        storage.guarded_remove("job", "../outside")


def test_guarded_remove_rejects_symlink(tmp_path: Path) -> None:
    _, storage, _, _ = setup(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = storage.job_root("link")
    storage.jobs_root.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this Windows host")

    with pytest.raises(UnsafeJobPathError):
        storage.guarded_remove("link")
