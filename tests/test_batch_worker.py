from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from bloodsmear.adapters.yolo11_onnx import AdapterTimings
from bloodsmear.batch.domain import (
    BatchItem,
    BatchJob,
    BatchMode,
    ItemStatus,
    JobStatus,
)
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage
from bloodsmear.batch.worker import BatchWorker
from bloodsmear.config import AppSettings
from bloodsmear.domain import (
    Detection,
    ImageInfo,
    InferenceResult,
    ModelInfo,
    QCInfo,
    TimingInfo,
    build_summary,
)


CLASSES = [
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelets",
    "RBC",
]
NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (40, 20), color=(255, 255, 255)).save(stream, format="PNG")
    return stream.getvalue()


def setup_job(
    tmp_path: Path,
    mode: BatchMode,
    filenames: list[str],
) -> tuple[BatchRepository, BatchStorage, AppSettings]:
    settings = AppSettings(
        database_path=tmp_path / "batch.db",
        jobs_root=tmp_path / "jobs",
    )
    repo = BatchRepository(settings.database_path)
    repo.initialize()
    storage = BatchStorage(settings)
    root = storage.job_root("job")
    (root / "inputs").mkdir(parents=True)
    items = []
    for ordinal, filename in enumerate(filenames):
        stored = f"item-{ordinal}.png"
        (root / "inputs" / stored).write_bytes(png_bytes())
        items.append(
            BatchItem(
                id=f"item-{ordinal}",
                job_id="job",
                ordinal=ordinal,
                original_filename=filename,
                stored_filename=stored,
                input_path=f"inputs/{stored}",
                sample_id="S001" if mode == BatchMode.GROUPED else f"sample-{ordinal}",
                view_id=f"view-{ordinal}",
                status=ItemStatus.PENDING,
            )
        )
    job = BatchJob(
        id="job",
        mode=mode,
        sample_id="S001" if mode == BatchMode.GROUPED else None,
        status=JobStatus.QUEUED,
        total_items=len(items),
        created_at=NOW,
        input_expires_at=NOW + timedelta(hours=24),
        result_expires_at=NOW + timedelta(days=7),
    )
    repo.create_job(job, items, lambda: None)
    return repo, storage, settings


class FakeInferenceService:
    def __init__(self, failing_filenames: set[str] | None = None) -> None:
        self.failing_filenames = failing_filenames or set()
        self.calls: list[str] = []

    def infer_bytes(
        self,
        data: bytes,
        filename: str,
        sample_id: str | None,
        options=None,
    ) -> InferenceResult:
        self.calls.append(filename)
        if filename in self.failing_filenames:
            raise RuntimeError(f"C:/secret/{filename}\ntraceback")
        detections = [
            Detection(
                class_id=6,
                class_name="RBC",
                confidence=0.9,
                bbox_xyxy=(1.0, 2.0, 15.0, 18.0),
            )
        ]
        summary, _ = build_summary(detections, CLASSES)
        return InferenceResult(
            sample_id=sample_id or "generated",
            created_at=NOW,
            model=ModelInfo(name="model", version="1", sha256="A" * 64),
            image=ImageInfo(
                filename=filename,
                width=40,
                height=20,
                sha256=hashlib.sha256(data).hexdigest().upper(),
            ),
            detections=detections,
            summary=summary,
            qc=QCInfo(status="pass", warnings=[]),
            runtime=TimingInfo(
                provider="CUDAExecutionProvider",
                preprocess_ms=1.0,
                inference_ms=2.0,
                postprocess_ms=1.0,
                total_ms=4.0,
            ),
        )


def worker(
    repo: BatchRepository,
    storage: BatchStorage,
    service: FakeInferenceService,
    report_service=None,
) -> BatchWorker:
    return BatchWorker(
        repo,
        storage,
        service,
        CLASSES,
        clock=lambda: NOW,
        poll_interval_seconds=0.01,
        report_service=report_service,
    )


def test_worker_processes_in_order_and_completes_with_zip(tmp_path: Path) -> None:
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["b.png", "a.png"])
    service = FakeInferenceService()
    subject = worker(repo, storage, service)

    assert subject.run_once() is True
    assert subject.run_once() is False

    detail = repo.get_job("job")
    assert service.calls == ["b.png", "a.png"]
    assert detail.status == JobStatus.COMPLETED
    assert detail.completed_items == 2
    summary = json.loads((storage.job_root("job") / "summary.json").read_text("utf-8"))
    assert summary["aggregate_summary"]["cell_counts"]["RBC"] == 2
    with zipfile.ZipFile(storage.job_root("job") / "results.zip") as archive:
        names = archive.namelist()
    assert names == sorted(names)
    assert "summary.json" in names
    assert "failures.json" in names
    assert "report.xlsx" in names
    assert "report.pdf" in names
    assert "items/001_b.png/result.json" in names
    assert "items/001_b.png/report.xlsx" in names
    assert "items/001_b.png/report.pdf" in names
    assert "items/002_a.png/report.pdf" in names
    assert (storage.job_root("job") / "items" / "item-0" / "report.xlsx").is_file()
    assert (storage.job_root("job") / "items" / "item-0" / "report.pdf").is_file()
    assert not any(name.startswith("inputs/") for name in names)


def test_worker_continues_after_failure_and_sanitizes_error(tmp_path: Path) -> None:
    repo, storage, _ = setup_job(
        tmp_path,
        BatchMode.GROUPED,
        ["bad.png", "good.png"],
    )
    service = FakeInferenceService({"bad.png"})

    worker(repo, storage, service).run_once()

    detail = repo.get_job("job")
    assert service.calls == ["bad.png", "good.png"]
    assert detail.status == JobStatus.PARTIAL_FAILED
    assert detail.failed_items == 1
    assert detail.items[0].error_code == "ITEM_INFERENCE_FAILED"
    assert "处理图片失败" in detail.items[0].error_message
    assert "secret" not in detail.items[0].error_message


def test_worker_marks_job_failed_when_all_items_fail(tmp_path: Path) -> None:
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["bad.png"])

    worker(repo, storage, FakeInferenceService({"bad.png"})).run_once()

    detail = repo.get_job("job")
    assert detail.status == JobStatus.FAILED
    summary = json.loads((storage.job_root("job") / "summary.json").read_text("utf-8"))
    assert all(
        value is None
        for value in summary["aggregate_summary"]["all_cell_ratios"].values()
    )


def test_report_failure_marks_partial_without_item_failure_and_keeps_base_zip(tmp_path: Path) -> None:
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["good.png"])

    class FailingReports:
        def generate_batch(self, *_args, **_kwargs):
            raise RuntimeError("private absolute path")

    worker(repo, storage, FakeInferenceService(), FailingReports()).run_once()

    detail = repo.get_job("job")
    assert detail.status == JobStatus.PARTIAL_FAILED
    assert detail.completed_items == 1
    assert detail.failed_items == 0
    assert detail.error_code == "REPORT_GENERATION_FAILED"
    assert (storage.job_root("job") / "results.zip").is_file()


def test_worker_skips_completed_item_after_recovery(tmp_path: Path) -> None:
    repo, storage, _ = setup_job(tmp_path, BatchMode.INDEPENDENT, ["done.png", "next.png"])
    repo.claim_next_job(NOW)
    repo.mark_item_running("item-0")
    result_dir = storage.job_root("job") / "items" / "item-0"
    result_dir.mkdir(parents=True)
    recovered_result = FakeInferenceService().infer_bytes(
        png_bytes(), "done.png", "sample-0"
    )
    (result_dir / "result.json").write_text(
        recovered_result.model_dump_json(), encoding="utf-8"
    )
    for name in ("detections.csv", "annotated.png"):
        (result_dir / name).write_bytes(b"existing")
    counts = {name: int(name == "RBC") for name in CLASSES}
    summary, _ = build_summary([], CLASSES)
    summary = summary.model_copy(
        update={
            "total_detected_cells": 1,
            "cell_counts": counts,
            "all_cell_ratios": {name: float(counts[name]) for name in CLASSES},
        }
    )
    repo.complete_item("item-0", "items/item-0", summary.model_dump_json(), 1.0)
    repo.mark_item_running("item-1")
    repo.recover_interrupted()
    service = FakeInferenceService()

    worker(repo, storage, service).run_once()

    assert service.calls == ["next.png"]
    assert repo.get_job("job").status == JobStatus.COMPLETED


def test_worker_does_not_infer_write_reports_or_finalize_when_item_claim_is_lost(tmp_path):
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["one.png"])
    with repo._connect() as connection:
        connection.execute("CREATE TRIGGER deny_item_claim BEFORE UPDATE OF status ON batch_items WHEN NEW.status = 'running' BEGIN SELECT RAISE(IGNORE); END")
    service = FakeInferenceService()
    assert worker(repo, storage, service).run_once() is True
    assert service.calls == []
    assert repo.get_job("job").status == JobStatus.RUNNING
    assert not (storage.job_root("job") / "items").exists()
    assert not (storage.job_root("job") / "results.zip").exists()


def test_worker_stop_finishes_current_image_but_does_not_start_remaining_images(tmp_path):
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["one.png", "two.png"])
    service = FakeInferenceService()
    subject = worker(repo, storage, service)
    original_infer = service.infer_bytes
    def infer_and_stop(*args, **kwargs):
        result = original_infer(*args, **kwargs)
        subject._stop_event.set()
        return result
    service.infer_bytes = infer_and_stop
    subject.run_once()
    assert service.calls == ["one.png"]
    assert repo.get_job("job").status == JobStatus.RUNNING
    assert repo.get_items("job")[0].status == ItemStatus.COMPLETED
    assert repo.get_items("job")[1].status == ItemStatus.PENDING
