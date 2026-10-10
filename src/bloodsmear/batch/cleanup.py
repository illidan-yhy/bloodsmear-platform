from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Thread
import logging

from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage


LOGGER = logging.getLogger("bloodsmear.batch.cleanup")


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
        self._cycle_failed = False

    def run_once(self, now: datetime | None = None) -> CleanupReport:
        current = now or datetime.now(timezone.utc)
        self._cycle_failed = False
        removed_jobs: set[str] = set()
        jobs_removed = 0
        for job in self.repository.list_expired_results(current):
            try:
                if self.storage.job_root(job.id).exists():
                    self.storage.guarded_remove(job.id)
                self.repository.delete_job(job.id)
                removed_jobs.add(job.id)
                jobs_removed += 1
            except Exception:
                self._cycle_failed = True
                LOGGER.exception("batch_cleanup_results_failed job_id=%s", job.id)

        inputs_removed = 0
        for job in self.repository.list_expired_inputs(current):
            if job.id in removed_jobs:
                continue
            try:
                inputs = self.storage.job_root(job.id) / "inputs"
                if inputs.exists():
                    self.storage.guarded_remove(job.id, "inputs")
                    inputs_removed += 1
                self.repository.mark_inputs_removed(job.id, current)
            except Exception:
                self._cycle_failed = True
                LOGGER.exception("batch_cleanup_inputs_failed job_id=%s", job.id)
        return CleanupReport(inputs_removed=inputs_removed, jobs_removed=jobs_removed)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run_loop, name="batch-cleanup", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                LOGGER.exception("batch_cleanup_cycle_failed")
                self._cycle_failed = True
            self._stop_event.wait(max(1.0, self.interval_seconds) if self._cycle_failed else self.interval_seconds)

    def stop(self, timeout: float = 30.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout)
