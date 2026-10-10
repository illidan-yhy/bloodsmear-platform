"""Failures must be actionable, logged, terminal when possible, and not kill loops."""
import logging
import sqlite3
from threading import Event, Thread

import pytest

from bloodsmear.batch.domain import BatchMode, JobStatus, ItemStatus
from bloodsmear.errors import InvalidImageError, GPUUnavailableError
from test_batch_worker import setup_job, worker, FakeInferenceService, NOW
from test_batch_cleanup import setup, add_job


@pytest.mark.parametrize(("error", "code", "message"), [
    (InvalidImageError("Unable to decode image data"), "INVALID_IMAGE", "损坏"),
    (InvalidImageError("Image pixel dimensions are too large"), "INVALID_IMAGE", "像素"),
    (InvalidImageError("Image exceeds the 25 MiB upload limit"), "INVALID_IMAGE", "大小"),
    (GPUUnavailableError("CUDA unavailable"), "GPU_UNAVAILABLE", "GPU"),
])
def test_item_keeps_specific_reason_and_exception_stack(tmp_path, caplog, error, code, message):
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["bad.png"])
    service = FakeInferenceService()
    def fail(*args, **kwargs):
        raise error
    service.infer_bytes = fail
    with caplog.at_level(logging.ERROR):
        worker(repo, storage, service).run_once()
    item = repo.get_items("job")[0]
    assert item.error_code == code and message in item.error_message
    assert any(record.exc_info for record in caplog.records)
    assert "job" in caplog.text and "item-0" in caplog.text


def test_zip_failure_preserves_successful_image_and_marks_batch_terminal(tmp_path, monkeypatch, caplog):
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["good.png"])
    def broken_zip(*args, **kwargs):
        raise OSError("private ZIP path")
    monkeypatch.setattr("bloodsmear.batch.worker.write_batch_artifacts", broken_zip)
    with caplog.at_level(logging.ERROR):
        worker(repo, storage, FakeInferenceService()).run_once()
    detail = repo.get_job("job")
    assert detail.status == JobStatus.PARTIAL_FAILED
    assert detail.error_code == "BATCH_PACKAGING_FAILED"
    assert detail.items[0].status == ItemStatus.COMPLETED
    assert (storage.job_root("job") / "items/item-0/result.json").is_file()
    assert any(record.exc_info for record in caplog.records)
    assert "private" not in detail.error_message


def test_summary_failure_marks_batch_terminal(tmp_path, monkeypatch):
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["good.png"])
    monkeypatch.setattr("bloodsmear.batch.worker.build_batch_summary", lambda *args: (_ for _ in ()).throw(ValueError("private summary")))
    worker(repo, storage, FakeInferenceService()).run_once()
    assert repo.get_job("job").error_code == "BATCH_SUMMARY_FAILED"
    assert repo.get_job("job").status == JobStatus.PARTIAL_FAILED


def test_worker_loop_survives_cycle_failure_and_logs_trace(tmp_path, caplog):
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["one.png"])
    subject = worker(repo, storage, FakeInferenceService())
    recovered = Event()
    calls = []
    def run_cycle():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("injected worker cycle failure")
        recovered.set()
        subject._stop_event.set()
        return False
    subject.run_once = run_cycle
    with caplog.at_level(logging.ERROR):
        thread = Thread(target=subject._run_loop)
        thread.start()
        try:
            assert recovered.wait(3), "Worker loop died instead of processing the next cycle"
        finally:
            subject._stop_event.set()
            thread.join(3)
    assert any(record.exc_info for record in caplog.records)


def test_cleanup_loop_survives_cycle_failure_and_logs_trace(tmp_path, caplog):
    _, _, _, subject = setup(tmp_path)
    recovered = Event()
    calls = []
    def run_cycle():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("injected cleanup cycle failure")
        recovered.set()
        subject._stop_event.set()
    subject.run_once = run_cycle
    with caplog.at_level(logging.ERROR):
        thread = Thread(target=subject._run_loop)
        thread.start()
        try:
            assert recovered.wait(3), "Cleanup loop died instead of retrying"
        finally:
            subject._stop_event.set()
            thread.join(3)
    assert any(record.exc_info for record in caplog.records)


def test_cleanup_skips_bad_job_and_still_cleans_sibling(tmp_path, monkeypatch, caplog):
    repo, storage, _, subject = setup(tmp_path)
    add_job(repo, storage, "bad", JobStatus.COMPLETED, input_expired=True, result_expired=True)
    add_job(repo, storage, "good", JobStatus.COMPLETED, input_expired=True, result_expired=True)
    remove = storage.guarded_remove
    def guarded_remove(job_id, *args):
        if job_id == "bad":
            raise PermissionError("injected cleanup permission error")
        return remove(job_id, *args)
    monkeypatch.setattr(storage, "guarded_remove", guarded_remove)
    with caplog.at_level(logging.ERROR):
        report = subject.run_once(NOW)
    assert report.jobs_removed == 1 and storage.job_root("bad").exists()
    assert not storage.job_root("good").exists()
    assert any(record.exc_info for record in caplog.records)


def test_failure_state_write_is_retried_without_running_model_again(tmp_path, monkeypatch):
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["one.png"])
    service = FakeInferenceService()
    subject = worker(repo, storage, service)
    monkeypatch.setattr("bloodsmear.batch.worker.write_batch_artifacts", lambda *args: (_ for _ in ()).throw(OSError("ZIP failure")))
    finish = repo.fail_job
    calls = []
    def fail_once(*args):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("temporary DB failure")
        return finish(*args)
    monkeypatch.setattr(repo, "fail_job", fail_once)
    with pytest.raises(sqlite3.OperationalError):
        subject.run_once()
    assert repo.get_job("job").status == JobStatus.RUNNING
    assert subject.run_once() is True
    assert repo.get_job("job").status == JobStatus.PARTIAL_FAILED
    assert service.calls == ["one.png"]


def test_exception_stack_is_written_to_real_app_log(tmp_path):
    from bloodsmear.logging_config import configure_logging, reset_managed_logging
    from bloodsmear.config import AppSettings
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["bad.png"])
    log_file = configure_logging(AppSettings(log_dir=tmp_path / "logs", log_to_console=False))
    try:
        worker(repo, storage, FakeInferenceService({"bad.png"})).run_once()
        text = log_file.read_text(encoding="utf-8")
        assert "batch_item_failed" in text and "Traceback" in text
        assert "job_id=job" in text and "item_id=item-0" in text
    finally:
        reset_managed_logging()


def test_next_queued_batch_completes_after_previous_zip_failure(tmp_path, monkeypatch):
    from test_batch_repository import make_job, make_items
    from test_batch_worker import png_bytes
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["one.png"])
    subject = worker(repo, storage, FakeInferenceService())
    pack = __import__("bloodsmear.batch.worker", fromlist=["write_batch_artifacts"]).write_batch_artifacts
    calls = []
    def pack_once_bad(*args):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("first ZIP failure")
        return pack(*args)
    monkeypatch.setattr("bloodsmear.batch.worker.write_batch_artifacts", pack_once_bad)
    subject.run_once()
    next_root = storage.job_root("next")
    (next_root / "inputs").mkdir(parents=True)
    (next_root / "inputs/item-0.png").write_bytes(png_bytes())
    next_item = make_items("next", 1)[0].model_copy(update={"id": "next-item"})
    repo.create_job(make_job("next", 1), [next_item], lambda: None)
    assert subject.run_once() is True
    assert repo.get_job("next").status == JobStatus.COMPLETED


def test_old_archive_is_not_served_after_packaging_failure(tmp_path, monkeypatch):
    from bloodsmear.batch.manager import BatchManager
    from bloodsmear.errors import JobNotReadyError
    repo, storage, settings = setup_job(tmp_path, BatchMode.GROUPED, ["one.png"])
    (storage.job_root("job") / "results.zip").write_bytes(b"stale archive")
    monkeypatch.setattr("bloodsmear.batch.worker.write_batch_artifacts", lambda *args: (_ for _ in ()).throw(OSError("ZIP failure")))
    worker(repo, storage, FakeInferenceService()).run_once()
    with pytest.raises(JobNotReadyError):
        BatchManager(repo, storage, settings).get_download_path("job")


def test_lost_claim_stops_before_later_exception_can_fail_someone_elses_work(tmp_path, monkeypatch):
    repo, storage, _ = setup_job(tmp_path, BatchMode.GROUPED, ["one.png", "two.png"])
    service = FakeInferenceService()
    claims = []
    def lose_then_fail(item_id):
        claims.append(item_id)
        if item_id == "item-0":
            return False
        raise sqlite3.OperationalError("later claim failure")
    monkeypatch.setattr(repo, "mark_item_running", lose_then_fail)
    worker(repo, storage, service).run_once()
    assert repo.get_job("job").status == JobStatus.RUNNING
    assert all(item.status == ItemStatus.PENDING for item in repo.get_items("job"))
    assert claims == ["item-0"] and service.calls == []
