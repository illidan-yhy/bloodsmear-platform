from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread

from bloodsmear.artifacts import (
    write_annotated_image,
    write_detections_csv,
    write_result_json,
)
from bloodsmear.batch.artifacts import write_batch_artifacts
from bloodsmear.batch.aggregation import build_batch_summary
from bloodsmear.batch.domain import ItemStatus, JobStatus
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage
from bloodsmear.domain import InferenceResult, QCInfo
from bloodsmear.reporting.context import ReportContextFactory
from bloodsmear.reporting.service import ReportService


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BatchWorker:
    def __init__(
        self,
        repository: BatchRepository,
        storage: BatchStorage,
        inference_service,
        classes: Sequence[str],
        *,
        clock: Callable[[], datetime] = _utc_now,
        poll_interval_seconds: float = 0.5,
        report_service: ReportService | None = None,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.inference_service = inference_service
        self.classes = list(classes)
        self.clock = clock
        self.poll_interval_seconds = poll_interval_seconds
        self.report_service = report_service or ReportService()
        self._stop_event = Event()
        self._thread: Thread | None = None

    def run_once(self) -> bool:
        job = self.repository.claim_next_job(self.clock())
        if job is None:
            return False
        job_root = self.storage.job_root(job.id)
        items = self.repository.get_items(job.id)
        report_error = False
        license_status = getattr(
            getattr(getattr(self.inference_service, "package", None), "manifest", None),
            "license_status",
            "unknown",
        )
        for item in items:
            if item.status != ItemStatus.PENDING:
                continue
            self.repository.mark_item_running(item.id)
            input_path = job_root / item.input_path
            try:
                image_data = input_path.read_bytes()
                infer_kwargs = {"sample_id": item.sample_id}
                if job.metadata is not None:
                    infer_kwargs["metadata"] = job.metadata
                result = self.inference_service.infer_bytes(
                    image_data, item.original_filename, **infer_kwargs
                )
                relative_result_dir = f"items/{item.id}"
                result_dir = job_root / relative_result_dir
                result_dir.mkdir(parents=True, exist_ok=True)
                write_result_json(result, result_dir / "result.json")
                write_detections_csv(result, result_dir / "detections.csv")
                write_annotated_image(
                    result,
                    image_data,
                    result_dir / "annotated.png",
                )
                try:
                    self.report_service.generate_single(
                        result, license_status, result_dir
                    )
                except Exception:
                    report_error = True
                    warnings = list(
                        dict.fromkeys(
                            [*result.qc.warnings, "REPORT_GENERATION_FAILED"]
                        )
                    )
                    result = result.model_copy(
                        update={"qc": QCInfo(status="warning", warnings=warnings)}
                    )
                    write_result_json(result, result_dir / "result.json")
                self.repository.complete_item(
                    item.id,
                    relative_result_dir,
                    result.summary.model_dump_json(),
                    result.runtime.inference_ms,
                )
            except Exception:
                self.repository.fail_item(
                    item.id,
                    "ITEM_INFERENCE_FAILED",
                    "Inference failed",
                )

        final_items = self.repository.get_items(job.id)
        completed = sum(item.status == ItemStatus.COMPLETED for item in final_items)
        failed = sum(item.status == ItemStatus.FAILED for item in final_items)
        if completed == len(final_items):
            final_status = JobStatus.COMPLETED
        elif failed == len(final_items):
            final_status = JobStatus.FAILED
        else:
            final_status = JobStatus.PARTIAL_FAILED
        artifact_job = job.model_copy(update={"status": final_status})
        batch_summary = build_batch_summary(artifact_job, final_items, self.classes)
        completed_results = [
            InferenceResult.model_validate_json(
                (job_root / item.result_directory / "result.json").read_text(encoding="utf-8")
            )
            for item in final_items
            if item.status == ItemStatus.COMPLETED and item.result_directory
        ]
        context = ReportContextFactory.batch(
            self.repository.get_job(job.id).model_copy(update={"status": final_status}),
            batch_summary,
            completed_results,
            license_status,
        )
        try:
            self.report_service.generate_batch(context, job_root)
        except Exception:
            report_error = True
            final_status = JobStatus.PARTIAL_FAILED
            artifact_job = artifact_job.model_copy(update={"status": final_status})
        if report_error:
            final_status = JobStatus.PARTIAL_FAILED
            artifact_job = artifact_job.model_copy(update={"status": final_status})
        write_batch_artifacts(artifact_job, final_items, job_root, self.classes)
        self.repository.finalize_job(
            job.id,
            final_status,
            self.clock(),
            error_code="REPORT_GENERATION_FAILED" if report_error else None,
            error_message="Report generation failed" if report_error else None,
        )
        return True

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.repository.recover_interrupted()
        self._stop_event.clear()
        self._thread = Thread(target=self._run_loop, name="batch-worker", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            processed = self.run_once()
            if not processed:
                self._stop_event.wait(self.poll_interval_seconds)

    def stop(self, timeout: float = 30.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout)
