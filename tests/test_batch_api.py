from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from bloodsmear.api import create_app
from bloodsmear.batch.domain import BatchItem, BatchJobDetail, BatchMode, ItemStatus, JobStatus
from bloodsmear.config import AppSettings
from bloodsmear.errors import GroupSampleIdRequiredError, JobNotFoundError, JobNotReadyError


NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def detail(status: JobStatus = JobStatus.QUEUED) -> BatchJobDetail:
    return BatchJobDetail(
        id="job-1",
        mode=BatchMode.GROUPED,
        sample_id="S001",
        status=status,
        total_items=1,
        completed_items=1 if status == JobStatus.COMPLETED else 0,
        failed_items=0,
        created_at=NOW,
        input_expires_at=NOW + timedelta(hours=24),
        result_expires_at=NOW + timedelta(days=7),
        items=[
            BatchItem(
                id="item-1",
                job_id="job-1",
                ordinal=0,
                original_filename="one.png",
                stored_filename="item.png",
                input_path="inputs/item.png",
                sample_id="S001",
                view_id="view-1",
                status=ItemStatus.COMPLETED if status == JobStatus.COMPLETED else ItemStatus.PENDING,
            )
        ],
    )


class FakeSingleService:
    adapter = SimpleNamespace(provider="CUDAExecutionProvider")
    package = SimpleNamespace(
        manifest=SimpleNamespace(
            name="model",
            version="1",
            sha256="A" * 64,
            classes=["RBC"],
            license_status="pending_publisher_confirmation",
        )
    )


class FakeBatchManager:
    def __init__(self, tmp_path: Path) -> None:
        self.current = detail()
        self.raise_on_create: Exception | None = None
        self.deleted: list[str] = []
        self.zip_path = tmp_path / "results.zip"
        self.zip_path.write_bytes(b"PK-test")

    async def create_job(self, mode, sample_id, files):
        if self.raise_on_create:
            raise self.raise_on_create
        assert mode == BatchMode.GROUPED
        assert sample_id == "S001"
        assert len(files) == 2
        return self.current

    def get_job(self, job_id: str):
        if job_id == "missing":
            raise JobNotFoundError("Job not found: missing")
        return self.current

    def get_results(self, job_id: str):
        if self.current.status == JobStatus.QUEUED:
            raise JobNotReadyError("Job is not ready")
        return {"job_id": job_id, "status": self.current.status.value}

    def get_download_path(self, job_id: str):
        if self.current.status == JobStatus.QUEUED:
            raise JobNotReadyError("Job is not ready")
        return self.zip_path

    def delete_job(self, job_id: str):
        self.deleted.append(job_id)


class LifecycleProbe:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


def make_client(tmp_path: Path):
    manager = FakeBatchManager(tmp_path)
    worker = LifecycleProbe()
    cleanup = LifecycleProbe()
    app = create_app(
        FakeSingleService(),
        AppSettings(output_dir=tmp_path / "outputs"),
        batch_manager=manager,
        batch_worker=worker,
        batch_cleanup=cleanup,
    )
    return app, manager, worker, cleanup


def test_create_batch_job_returns_202_and_links(tmp_path: Path) -> None:
    app, _, _, _ = make_client(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/jobs",
            data={"mode": "grouped", "sample_id": "S001"},
            files=[
                ("files", ("one.png", b"one", "image/png")),
                ("files", ("two.png", b"two", "image/png")),
            ],
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["job_id"] == "job-1"
    assert payload["status_url"] == "/api/v1/jobs/job-1"
    assert payload["results_url"] == "/api/v1/jobs/job-1/results"
    assert payload["download_url"] == "/api/v1/jobs/job-1/download"


def test_batch_create_maps_validation_error(tmp_path: Path) -> None:
    app, manager, _, _ = make_client(tmp_path)
    manager.raise_on_create = GroupSampleIdRequiredError("sample required")
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/jobs",
            data={"mode": "grouped"},
            files=[("files", ("one.png", b"one", "image/png"))],
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "GROUP_SAMPLE_ID_REQUIRED"


def test_batch_status_and_unknown_job(tmp_path: Path) -> None:
    app, _, _, _ = make_client(tmp_path)
    with TestClient(app) as client:
        status = client.get("/api/v1/jobs/job-1")
        missing = client.get("/api/v1/jobs/missing")
    assert status.status_code == 200
    assert status.json()["total_items"] == 1
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_results_and_download_require_terminal_job(tmp_path: Path) -> None:
    app, _, _, _ = make_client(tmp_path)
    with TestClient(app) as client:
        results = client.get("/api/v1/jobs/job-1/results")
        download = client.get("/api/v1/jobs/job-1/download")
    assert results.status_code == 409
    assert download.status_code == 409
    assert results.json()["error"]["code"] == "JOB_NOT_READY"


def test_completed_results_download_and_delete(tmp_path: Path) -> None:
    app, manager, _, _ = make_client(tmp_path)
    manager.current = detail(JobStatus.COMPLETED)
    with TestClient(app) as client:
        results = client.get("/api/v1/jobs/job-1/results")
        download = client.get("/api/v1/jobs/job-1/download")
        deleted = client.delete("/api/v1/jobs/job-1")
    assert results.status_code == 200
    assert results.json()["status"] == "completed"
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert deleted.status_code == 204
    assert manager.deleted == ["job-1"]


def test_lifespan_starts_and_stops_worker_and_cleanup(tmp_path: Path) -> None:
    app, _, worker, cleanup = make_client(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert worker.started == 1
        assert cleanup.started == 1
    assert worker.stopped == 1
    assert cleanup.stopped == 1
