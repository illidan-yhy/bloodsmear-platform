from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
import logging
from threading import Event, Thread, Lock

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
from bloodsmear.service_instance import ServiceLock
from bloodsmear.errors import ServiceAlreadyRunningError, JobNotFoundError, batch_item_error


LOGGER = logging.getLogger("bloodsmear.batch.worker")


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
        service_lock: ServiceLock | None = None,
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
        self._service_lock = service_lock
        self._owns_service_lock = service_lock is None
        self._lifecycle_lock = Lock()
        self._pending_failure: tuple[str, str, str] | None = None
        self._stage = "processing"

    def run_once(self) -> bool:
        if self._stop_event.is_set():
            return False
        if self._pending_failure is not None:
            self._persist_failure()
            return True
        job = self.repository.claim_next_job(self.clock())
        if job is None:
            return False
        self._stage = "processing"
        try:
            return self._process_job(job)
        except Exception:
            code, message = {
                "summary": ("BATCH_SUMMARY_FAILED", "批量汇总失败，已成功的图片结果保留，请查看日志。"),
                "packaging": ("BATCH_PACKAGING_FAILED", "批量打包失败，已成功的图片结果保留，请查看日志。"),
            }.get(self._stage, ("BATCH_PROCESSING_FAILED", "批量处理异常，已生成的结果文件保留，请查看日志。"))
            LOGGER.exception("batch_job_failed job_id=%s stage=%s error_code=%s", job.id, self._stage, code)
            self._pending_failure = (job.id, code, message)
            self._persist_failure()
            return True

    def _persist_failure(self) -> None:
        job_id, code, message = self._pending_failure
        try:
            self.repository.fail_job(job_id, self.clock(), code, message)
        except JobNotFoundError:
            LOGGER.warning("batch_failure_job_removed job_id=%s", job_id)
        self._pending_failure = None

    def _process_job(self, job) -> bool:
        job_root = self.storage.job_root(job.id)
        items = self.repository.get_items(job.id)
        report_error = False
        license_status = getattr(
            getattr(getattr(self.inference_service, "package", None), "manifest", None),
            "license_status",
            "unknown",
        )
        for item in items:
            if self._stop_event.is_set():
                return True
            if item.status != ItemStatus.PENDING:
                continue
            if not self.repository.mark_item_running(item.id):
                LOGGER.warning("batch_item_claim_lost job_id=%s item_id=%s", job.id, item.id)
                # Ownership is uncertain; do not process siblings or terminate this job.
                return True
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
                    LOGGER.exception("batch_item_report_failed job_id=%s item_id=%s", job.id, item.id)
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
            except Exception as exc:
                code, message = batch_item_error(exc)
                LOGGER.exception("batch_item_failed job_id=%s item_id=%s error_code=%s", job.id, item.id, code)
                self.repository.fail_item(
                    item.id,
                    code,
                    message,
                )

        if self._stop_event.is_set():
            return True
        final_items = self.repository.get_items(job.id)
        if any(item.status in (ItemStatus.PENDING, ItemStatus.RUNNING) for item in final_items):
            return True
        completed = sum(item.status == ItemStatus.COMPLETED for item in final_items)
        failed = sum(item.status == ItemStatus.FAILED for item in final_items)
        if completed == len(final_items):
            final_status = JobStatus.COMPLETED
        elif failed == len(final_items):
            final_status = JobStatus.FAILED
        else:
            final_status = JobStatus.PARTIAL_FAILED
        self._stage = "summary"
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
            LOGGER.exception("batch_report_failed job_id=%s", job.id)
            report_error = True
            final_status = JobStatus.PARTIAL_FAILED
            artifact_job = artifact_job.model_copy(update={"status": final_status})
        if report_error:
            final_status = JobStatus.PARTIAL_FAILED
            artifact_job = artifact_job.model_copy(update={"status": final_status})
        self._stage = "packaging"
        write_batch_artifacts(artifact_job, final_items, job_root, self.classes)
        self._stage = "finalization"
        self.repository.finalize_job(
            job.id,
            final_status,
            self.clock(),
            error_code="REPORT_GENERATION_FAILED" if report_error else None,
            error_message="部分报告生成失败，基础结果保留，请查看日志。" if report_error else None,
        )
        return True

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread and self._thread.is_alive():
                return
            if self._service_lock is None:
                self._service_lock = ServiceLock(self.repository.database_path)
            if self._owns_service_lock and not self._service_lock.acquire():
                raise ServiceAlreadyRunningError("同一任务数据库的服务已打开或处理实例已在运行。")
            if not self._service_lock.acquired:
                raise ServiceAlreadyRunningError("启动保护尚未取得，不能恢复或处理任务。")
            try:
                self.repository.recover_interrupted()
                self._stop_event.clear()
                self._thread = Thread(target=self._run_loop, name="batch-worker", daemon=True)
                self._thread.start()
            except BaseException:
                if self._owns_service_lock:
                    self._service_lock.release()
                raise

    def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    processed = self.run_once()
                except Exception:
                    LOGGER.exception("batch_worker_cycle_failed")
                    self._stop_event.wait(max(1.0, self.poll_interval_seconds))
                    continue
                if not processed:
                    self._stop_event.wait(self.poll_interval_seconds)
        finally:
            if self._owns_service_lock and self._service_lock is not None:
                self._service_lock.release()

    def stop(self, timeout: float | None = 30.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout)
