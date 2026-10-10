"""Run the actual status script in Windows PowerShell 5.1 and PowerShell 7."""
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading

import pytest


@pytest.fixture(params=["powershell.exe", "pwsh.exe"])
def shell(request):
    executable = shutil.which(request.param)
    if os.name != "nt" or executable is None:
        pytest.skip(f"{request.param} is unavailable")
    return executable


@contextmanager
def endpoint(status, payload):
    calls = []
    body = json.dumps(payload).encode("utf-8") if isinstance(payload, dict) else payload.encode("utf-8")
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_response(status if self.path == "/health/ready" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def run_status(tmp_path, shell, port=None, record=None, dotenv="", env_settings=None, log_folder="logs"):
    package = tmp_path / "package with spaces"
    package.mkdir(exist_ok=True)
    script = package / "status.ps1"
    shutil.copyfile(Path(__file__).parents[1] / "deployment/windows/status.ps1", script)
    if dotenv:
        (package / ".env").write_text(dotenv, encoding="utf-8")
    if record is not None:
        directory = package / log_folder
        directory.mkdir(parents=True, exist_ok=True)
        data = json.dumps(record) if isinstance(record, dict) else record
        (directory / "startup-status.json").write_text(data, encoding="utf-8")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("BLOODSMEAR_")}
    if Path(shell).name.lower() == "powershell.exe":
        environment["PSModulePath"] = str(Path(environment.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/Modules")
    environment.update(env_settings or {})
    command = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    if port is not None:
        command.extend(["-Port", str(port)])
    completed = subprocess.run(command, capture_output=True, timeout=20, env=environment)
    return completed.returncode, (completed.stdout + completed.stderr).decode("utf-8", errors="replace")


def gpu_failure(port):
    return {
        "status": "failed", "error_code": "GPU_UNAVAILABLE", "provider": "CPUExecutionProvider",
        "port": port, "require_gpu": True, "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def test_status_reports_live_gpu_failure_not_stopped_service(tmp_path, shell):
    with endpoint(503, {"error": {"code": "GPU_UNAVAILABLE", "message": "private driver path"}}) as (port, calls):
        code, output = run_status(tmp_path, shell, port)
    assert code == 2
    assert "GPU 不可用" in output and "驱动" in output and "VC++" in output
    assert "服务未运行" not in output and "private" not in output
    assert calls == ["/health/ready"]


@pytest.mark.parametrize(("status", "payload", "expected"), [
    (503, {"error": {"code": "MODEL_NOT_FOUND", "message": "private model path"}}, "MODEL_NOT_FOUND"),
    (500, "private traceback", "HTTP 500"),
])
def test_status_does_not_label_all_http_errors_as_gpu_failure(tmp_path, shell, status, payload, expected):
    with endpoint(status, payload) as (port, _):
        code, output = run_status(tmp_path, shell, port)
    assert code == 3
    assert expected in output and "GPU 不可用" not in output and "private" not in output


@pytest.mark.parametrize("provider", ["CUDAExecutionProvider", "CPUExecutionProvider"])
def test_live_readiness_takes_precedence_over_old_failure_record(tmp_path, shell, provider):
    with endpoint(200, {"status": "ready", "provider": provider}) as (port, _):
        code, output = run_status(tmp_path, shell, port, record=gpu_failure(port))
    assert code == 0
    assert ("GPU 正常" if provider == "CUDAExecutionProvider" else "CPU 模式") in output
    assert "GPU 不可用" not in output


def test_status_distinguishes_stopped_service_without_failure_record(tmp_path, shell):
    code, output = run_status(tmp_path, shell, unused_port())
    assert code == 1 and "服务未运行" in output
    assert "GPU 不可用" not in output


def test_stopped_service_explains_last_gpu_startup_failure(tmp_path, shell):
    port = unused_port()
    code, output = run_status(tmp_path, shell, port, record=gpu_failure(port))
    assert code == 2
    assert "服务未运行" in output and "最近一次启动" in output and "GPU 不可用" in output


@pytest.mark.parametrize("record_kind", ["different_port", "normal_stop", "corrupt"])
def test_status_ignores_inapplicable_or_invalid_gpu_record(tmp_path, shell, record_kind):
    port = unused_port()
    record = gpu_failure(port)
    if record_kind == "different_port":
        record["port"] += 1
    elif record_kind == "normal_stop":
        record.update(status="stopped", error_code=None)
    else:
        record = "not valid JSON"
    code, output = run_status(tmp_path, shell, port, record=record)
    assert code == 1 and "服务未运行" in output
    assert "GPU 不可用" not in output


def test_status_reads_port_from_dotenv(tmp_path, shell):
    with endpoint(200, {"status": "ready", "provider": "CUDAExecutionProvider"}) as (port, calls):
        code, output = run_status(tmp_path, shell, dotenv=f'BLOODSMEAR_PORT="{port}"\n')
    assert code == 0 and "GPU 正常" in output
    assert calls == ["/health/ready"]


def test_status_environment_port_overrides_dotenv(tmp_path, shell):
    with endpoint(200, {"status": "ready", "provider": "CUDAExecutionProvider"}) as (port, calls):
        code, output = run_status(tmp_path, shell, dotenv='BLOODSMEAR_PORT=1\n', env_settings={"BLOODSMEAR_PORT": str(port)})
    assert code == 0 and "GPU 正常" in output
    assert calls == ["/health/ready"]


def test_status_reads_custom_log_directory_from_dotenv(tmp_path, shell):
    port = unused_port()
    code, output = run_status(
        tmp_path, shell, port, record=gpu_failure(port),
        dotenv='BLOODSMEAR_LOG_DIR="custom logs"\n', log_folder="custom logs",
    )
    assert code == 2 and "最近一次启动" in output and "GPU 不可用" in output


def test_status_uses_last_dotenv_value_like_the_application(tmp_path, shell):
    with endpoint(200, {"status": "ready", "provider": "CUDAExecutionProvider"}) as (port, calls):
        code, output = run_status(tmp_path, shell, dotenv=f'BLOODSMEAR_PORT=1\nBLOODSMEAR_PORT={port}\n')
    assert code == 0 and "GPU 正常" in output
    assert calls == ["/health/ready"]


def test_status_reports_non_gpu_startup_failure_after_service_exits(tmp_path, shell):
    port = unused_port()
    record = gpu_failure(port)
    record["error_code"] = "MODEL_NOT_FOUND"
    code, output = run_status(tmp_path, shell, port, record=record)
    assert code == 3 and "MODEL_NOT_FOUND" in output
    assert "GPU 不可用" not in output


def test_status_environment_log_directory_overrides_dotenv(tmp_path, shell):
    port = unused_port()
    code, output = run_status(
        tmp_path, shell, port, record=gpu_failure(port),
        dotenv='BLOODSMEAR_LOG_DIR="other logs"\n',
        env_settings={"BLOODSMEAR_LOG_DIR": "environment logs"}, log_folder="environment logs",
    )
    assert code == 2 and "GPU 不可用" in output


def test_status_does_not_change_the_callers_process_proxy(tmp_path, shell):
    script = tmp_path / "status.ps1"
    shutil.copyfile(Path(__file__).parents[1] / "deployment/windows/status.ps1", script)
    environment = os.environ.copy()
    if Path(shell).name.lower() == "powershell.exe":
        environment["PSModulePath"] = str(Path(environment.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/Modules")
    with endpoint(200, {"status": "ready", "provider": "CUDAExecutionProvider"}) as (port, calls):
        command = (
            "$originalProxy = [Net.WebProxy]::new('http://127.0.0.1:1'); "
            "[Net.WebRequest]::DefaultWebProxy = $originalProxy; "
            f"& '{str(script).replace(chr(39), chr(39) * 2)}' -Port {port}; "
            "if (-not [object]::ReferenceEquals($originalProxy, [Net.WebRequest]::DefaultWebProxy)) { exit 9 }; exit 0"
        )
        completed = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command], capture_output=True, timeout=20, env=environment)
    assert completed.returncode == 0, (completed.stdout + completed.stderr).decode("utf-8", errors="replace")
    assert calls == ["/health/ready"]


@pytest.mark.parametrize("quote", ['"', "'"])
def test_status_accepts_quoted_port_followed_by_comment(tmp_path, shell, quote):
    with endpoint(200, {"status": "ready", "provider": "CUDAExecutionProvider"}) as (port, calls):
        code, output = run_status(tmp_path, shell, dotenv=f'BLOODSMEAR_PORT={quote}{port}{quote} # local port\n')
    assert code == 0 and "GPU 正常" in output
    assert calls == ["/health/ready"]


def test_status_preserves_hash_inside_quoted_log_directory(tmp_path, shell):
    port = unused_port()
    code, output = run_status(
        tmp_path, shell, port, record=gpu_failure(port),
        dotenv='BLOODSMEAR_LOG_DIR="custom # logs" # diagnostics\n', log_folder="custom # logs",
    )
    assert code == 2 and "GPU 不可用" in output
