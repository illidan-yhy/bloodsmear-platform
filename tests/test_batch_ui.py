from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from bloodsmear.api import create_app
from bloodsmear.batch.domain import BatchJobDetail, BatchMode, JobStatus
from bloodsmear.config import AppSettings


NOW = datetime(2026, 10, 4, tzinfo=timezone.utc)


def job(status: JobStatus) -> BatchJobDetail:
    return BatchJobDetail(
        id="job-1",
        mode=BatchMode.GROUPED,
        sample_id="S001",
        status=status,
        total_items=2,
        completed_items=2 if status == JobStatus.COMPLETED else 0,
        failed_items=0,
        created_at=NOW,
        input_expires_at=NOW + timedelta(hours=24),
        result_expires_at=NOW + timedelta(days=7),
        items=[],
    )


class SingleService:
    adapter = SimpleNamespace(provider="CUDAExecutionProvider")
    package = SimpleNamespace(
        manifest=SimpleNamespace(
            name="model", version="1", sha256="A" * 64,
            classes=["RBC"], license_status="pending_publisher_confirmation"
        )
    )


class Manager:
    def __init__(self) -> None:
        self.current = job(JobStatus.QUEUED)

    async def create_job(self, mode, sample_id, files):
        return self.current

    def get_job(self, job_id):
        return self.current


def test_home_renders_batch_modes_multi_file_and_no_cdn(tmp_path: Path) -> None:
    app = create_app(SingleService(), AppSettings(output_dir=tmp_path), Manager())
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert 'value="independent"' in response.text
    assert 'value="grouped"' in response.text
    assert 'name="files"' in response.text
    assert "multiple" in response.text
    assert 'name="sample_id"' in response.text
    assert "https://unpkg.com" not in response.text


def test_batch_submission_returns_polling_fragment(tmp_path: Path) -> None:
    app = create_app(SingleService(), AppSettings(output_dir=tmp_path), Manager())
    with TestClient(app) as client:
        response = client.post(
            "/ui/jobs",
            data={"mode": "grouped", "sample_id": "S001"},
            files=[
                ("files", ("one.png", b"one", "image/png")),
                ("files", ("two.png", b"two", "image/png")),
            ],
        )
    assert response.status_code == 202
    assert 'hx-get="/ui/jobs/job-1/status"' in response.text
    assert 'hx-trigger="every 2s"' in response.text


def test_running_status_keeps_polling_and_terminal_status_stops(tmp_path: Path) -> None:
    manager = Manager()
    app = create_app(SingleService(), AppSettings(output_dir=tmp_path), manager)
    with TestClient(app) as client:
        running = client.get("/ui/jobs/job-1/status")
        manager.current = job(JobStatus.COMPLETED)
        completed = client.get("/ui/jobs/job-1/status")
    assert 'hx-trigger="every 2s"' in running.text
    assert "hx-trigger" not in completed.text
    assert "/api/v1/jobs/job-1/download" in completed.text
    assert "下载 ZIP" in completed.text
