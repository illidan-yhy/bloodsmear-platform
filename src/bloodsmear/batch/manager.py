from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from bloodsmear.batch.domain import (
    BatchItem,
    BatchJob,
    BatchJobDetail,
    BatchMode,
    ItemStatus,
    JobStatus,
    TERMINAL_JOB_STATUSES,
)
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import AsyncUpload, BatchStorage, safe_stem
from bloodsmear.config import AppSettings
from bloodsmear.domain import SampleMetadata
from bloodsmear.errors import (
    BatchTooManyFilesError,
    GroupSampleIdRequiredError,
    InvalidImageError,
    JobNotReadyError,
)
import json
from pathlib import Path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BatchManager:
    def __init__(
        self,
        repository: BatchRepository,
        storage: BatchStorage,
        settings: AppSettings,
        *,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.settings = settings
        self.clock = clock
        self.id_factory = id_factory

    async def create_job(
        self,
        mode: BatchMode,
        sample_id: str | None,
        files: Sequence[AsyncUpload],
        metadata: SampleMetadata | None = None,
    ) -> BatchJobDetail:
        if not files:
            raise InvalidImageError("A batch must contain at least one image")
        if len(files) > self.settings.batch_max_files:
            raise BatchTooManyFilesError(
                f"A batch may contain at most {self.settings.batch_max_files} images"
            )
        normalized_sample_id = sample_id.strip() if sample_id else None
        if mode == BatchMode.GROUPED and not normalized_sample_id:
            raise GroupSampleIdRequiredError(
                "sample_id is required for grouped mode"
            )

        job_id = self.id_factory()
        staged = await self.storage.stage_uploads(job_id, files)
        now = self.clock()
        job = BatchJob(
            id=job_id,
            mode=mode,
            sample_id=normalized_sample_id,
            metadata=metadata,
            status=JobStatus.QUEUED,
            total_items=len(staged),
            created_at=now,
            input_expires_at=now + timedelta(hours=self.settings.input_retention_hours),
            result_expires_at=now + timedelta(days=self.settings.result_retention_days),
        )
        items: list[BatchItem] = []
        for upload in staged:
            stem = safe_stem(upload.original_filename)
            item_sample_id = (
                normalized_sample_id
                if mode == BatchMode.GROUPED
                else f"{stem}-{upload.item_id[:8]}"
            )
            items.append(
                BatchItem(
                    id=upload.item_id,
                    job_id=job_id,
                    ordinal=upload.ordinal,
                    original_filename=upload.original_filename,
                    stored_filename=upload.stored_filename,
                    input_path=upload.input_path,
                    sample_id=item_sample_id,
                    view_id=f"{stem}-{upload.ordinal + 1}-{upload.item_id[:8]}",
                    status=ItemStatus.PENDING,
                )
            )
        try:
            self.repository.create_job(
                job,
                items,
                lambda: self.storage.promote(job_id),
            )
        except Exception:
            self.storage.discard_staging(job_id)
            if self.storage.job_root(job_id).exists():
                self.storage.guarded_remove(job_id)
            raise
        return self.repository.get_job(job_id)

    def get_job(self, job_id: str) -> BatchJobDetail:
        return self.repository.get_job(job_id)

    def delete_job(self, job_id: str) -> None:
        job = self.repository.get_job(job_id)
        if job.status not in TERMINAL_JOB_STATUSES:
            raise JobNotReadyError("Only terminal jobs can be deleted")
        if self.storage.job_root(job_id).exists():
            self.storage.guarded_remove(job_id)
        self.repository.delete_job(job_id)

    def get_results(self, job_id: str) -> dict:
        job = self.repository.get_job(job_id)
        if job.status not in TERMINAL_JOB_STATUSES:
            raise JobNotReadyError("Job is not ready")
        path = self.storage.job_root(job_id) / "summary.json"
        if not path.is_file():
            raise JobNotReadyError("Job summary is not ready")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_download_path(self, job_id: str) -> Path:
        job = self.repository.get_job(job_id)
        if job.status not in TERMINAL_JOB_STATUSES:
            raise JobNotReadyError("Job is not ready")
        path = self.storage.job_root(job_id) / "results.zip"
        if not path.is_file():
            raise JobNotReadyError("Job archive is not ready")
        return path
