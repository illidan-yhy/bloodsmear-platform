from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Thread

from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage


@dataclass(frozen=True)
class CleanupReport:
    inputs_removed: int
    jobs_removed: int


class BatchCleanup:
    def __init__(
        self,
        repository: BatchRepository,
        storage: BatchStorage,
        *,
        interval_seconds: float,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.interval_seconds = interval_seconds
        self._stop_event = Event()
        self._thread: Thread | None = None

    def run_once(self, now: datetime | None = None) -> CleanupReport:
        current = now or datetime.now(timezone.utc)
        removed_jobs: set[str] = set()
        jobs_removed = 0
        for job in self.repository.list_expired_results(current):
            if self.storage.job_root(job.id).exists():
                self.storage.guarded_remove(job.id)
            self.repository.delete_job(job.id)
            removed_jobs.add(job.id)
            jobs_removed += 1

        inputs_removed = 0
        for job in self.repository.list_expired_inputs(current):
            if job.id in removed_jobs:
                continue
            inputs = self.storage.job_root(job.id) / "inputs"
            if inputs.exists():
                self.storage.guarded_remove(job.id, "inputs")
                inputs_removed += 1
            self.repository.mark_inputs_removed(job.id, current)
        return CleanupReport(inputs_removed=inputs_removed, jobs_removed=jobs_removed)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run_loop, name="batch-cleanup", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            self._stop_event.wait(self.interval_seconds)

    def stop(self, timeout: float = 30.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout)
