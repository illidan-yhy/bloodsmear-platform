"""Safe temporary-database and process tests for duplicate-start protection."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from bloodsmear.cli import main
from test_cli import serve_settings, FakeService, install_fake_service


@contextmanager
def platform_endpoint(live_payload=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = (live_payload if live_payload is not None else {"status": "alive"}) if self.path == "/health/live" else {
                "name": "model", "version": "1", "sha256": "A" * 64, "classes": ["RBC"],
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_duplicate_service_opens_existing_browser_without_loading_model_or_changing_record(tmp_path, monkeypatch, capsys):
    with platform_endpoint() as port:
        settings = serve_settings(tmp_path, False).model_copy(update={"host": "127.0.0.1", "port": port})
        settings.log_dir.mkdir(parents=True)
        record = settings.log_dir / "startup-status.json"
        record.write_text('{"status":"starting","provider":"CUDAExecutionProvider"}', encoding="utf-8")
        before = record.read_bytes()
        monkeypatch.setattr("bloodsmear.cli.AppSettings", lambda: settings)
        model_calls = []
        monkeypatch.setattr("bloodsmear.cli.InferenceService.from_settings", lambda settings: model_calls.append(settings))
        browser_calls = []
        import webbrowser
        monkeypatch.setattr(webbrowser, "open", lambda url: browser_calls.append(url))
        assert main(["serve", "--open-browser"]) == 0
        assert model_calls == [] and record.read_bytes() == before
        assert not settings.database_path.exists()
        assert browser_calls == [f"http://127.0.0.1:{port}"]
        assert "服务已打开" in capsys.readouterr().out


def test_other_occupied_port_does_not_load_model_or_recover_tasks(tmp_path, monkeypatch, capsys):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        settings = serve_settings(tmp_path, False).model_copy(update={"host": "127.0.0.1", "port": listener.getsockname()[1]})
        monkeypatch.setattr("bloodsmear.cli.AppSettings", lambda: settings)
        install_fake_service(monkeypatch, FakeService())
        server_calls = []
        monkeypatch.setattr("bloodsmear.cli.uvicorn.run", lambda *args, **kwargs: server_calls.append(args))
        assert main(["serve"]) == 1
        assert server_calls == [] and not settings.database_path.exists()
        assert "端口" in capsys.readouterr().err


def test_process_lock_prevents_another_process_and_releases_after_crash(tmp_path):
    from bloodsmear.service_instance import ServiceLock
    database = tmp_path / "jobs.db"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    code = "from pathlib import Path; import sys,time; from bloodsmear.service_instance import ServiceLock; lock=ServiceLock(Path(sys.argv[1]),port=8123); lock.acquire(); print('locked',flush=True); time.sleep(30)"
    child = subprocess.Popen([sys.executable, "-u", "-c", code, str(database)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    try:
        assert child.stdout.readline().strip() == b"locked"
        duplicate = ServiceLock(database, port=8124)
        assert duplicate.acquire() is False
        child.kill()
        child.wait(timeout=5)
        assert duplicate.acquire() is True
        duplicate.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_second_worker_does_not_recover_running_job(tmp_path):
    from bloodsmear.service_instance import ServiceLock
    from test_batch_worker import setup_job, worker, FakeInferenceService, NOW
    from bloodsmear.batch.domain import BatchMode, ItemStatus, JobStatus
    repo, storage, settings = setup_job(tmp_path, BatchMode.GROUPED, ["one.png"])
    repo.claim_next_job(NOW)
    repo.mark_item_running("item-0")
    lock = ServiceLock(settings.database_path, port=8123)
    assert lock.acquire()
    try:
        subject = worker(repo, storage, FakeInferenceService())
        with pytest.raises(Exception, match="已打开|已在运行"):
            subject.start()
        detail = repo.get_job("job")
        assert detail.status == JobStatus.RUNNING and detail.items[0].status == ItemStatus.RUNNING
    finally:
        lock.release()


def test_real_second_cli_process_keeps_active_batch_and_opens_existing_url(tmp_path):
    from test_batch_worker import setup_job, NOW
    from bloodsmear.batch.domain import BatchMode, ItemStatus, JobStatus
    repo, storage, settings = setup_job(tmp_path, BatchMode.GROUPED, ["one.png"])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {key: value for key, value in os.environ.items() if not key.startswith("BLOODSMEAR_")}
    environment.update({
        "PYTHONPATH": os.pathsep.join([str(Path(__file__).parents[1] / "src"), str(Path(__file__).parent)]),
        "BLOODSMEAR_HOST": "127.0.0.1", "BLOODSMEAR_PORT": str(port),
        "BLOODSMEAR_DATABASE_PATH": str(settings.database_path), "BLOODSMEAR_JOBS_ROOT": str(settings.jobs_root),
        "BLOODSMEAR_OUTPUT_DIR": str(tmp_path / "outputs"), "BLOODSMEAR_LOG_DIR": str(tmp_path / "logs"),
        "BLOODSMEAR_REQUIRE_GPU": "false", "BLOODSMEAR_LOG_TO_CONSOLE": "false",
    })
    command = [sys.executable, "-u", str(Path(__file__).with_name("single_instance_child.py")), str(tmp_path)]
    first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    try:
        deadline = time.monotonic() + 12
        while not (tmp_path / "image-started").exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.03)
        assert (tmp_path / "image-started").exists(), "First process did not reach model execution"
        before = (tmp_path / "logs/startup-status.json").read_bytes()
        started = time.monotonic()
        second = subprocess.run(command + ["--open-browser"], capture_output=True, timeout=10, env=environment)
        assert second.returncode == 0, (second.stdout + second.stderr).decode("utf-8", errors="replace")
        assert time.monotonic() - started < 10
        assert "服务已打开" in second.stdout.decode("utf-8", errors="replace")
        assert (tmp_path / "opened-browser.txt").read_text(encoding="utf-8") == f"http://127.0.0.1:{port}"
        assert (tmp_path / "logs/startup-status.json").read_bytes() == before
        detail = repo.get_job("job")
        assert detail.status == JobStatus.RUNNING and detail.items[0].status == ItemStatus.RUNNING
        assert (tmp_path / "model-calls.txt").read_text().splitlines() == ["infer"]
        (tmp_path / "release-image").touch()
        deadline = time.monotonic() + 8
        while repo.get_job("job").status == JobStatus.RUNNING and time.monotonic() < deadline:
            time.sleep(0.03)
        assert repo.get_job("job").status == JobStatus.COMPLETED
    finally:
        (tmp_path / "release-image").touch()
        if first.poll() is None:
            first.terminate()
        first.wait(timeout=5)


def test_reserved_port_is_already_listening_before_model_or_lifespan_start():
    from bloodsmear.service_instance import reserve_port
    listener = reserve_port("127.0.0.1", 0)
    try:
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    finally:
        listener.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX SO_REUSEADDR reservation race")
def test_posix_reserved_port_cannot_be_reserved_by_a_second_instance():
    from bloodsmear.service_instance import reserve_port
    listener = reserve_port("127.0.0.1", 0)
    try:
        with pytest.raises(OSError):
            second = reserve_port("127.0.0.1", listener.getsockname()[1])
            second.close()
    finally:
        listener.close()


def test_unrelated_json_service_is_not_treated_as_existing_platform(tmp_path, monkeypatch, capsys):
    with platform_endpoint(["not-the-platform"]) as port:
        settings = serve_settings(tmp_path, False).model_copy(update={"host": "127.0.0.1", "port": port})
        monkeypatch.setattr("bloodsmear.cli.AppSettings", lambda: settings)
        assert main(["serve", "--open-browser"]) == 1
        assert "端口" in capsys.readouterr().err
        assert not settings.database_path.exists()
