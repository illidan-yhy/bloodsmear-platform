from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_python_311_non_root_and_does_not_bake_model() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM python:3.11-slim")
    assert "requirements.lock" in dockerfile
    assert "USER app" in dockerfile
    assert "COPY models" not in dockerfile
    assert "model.onnx" not in dockerfile


def test_compose_mounts_model_read_only_and_requests_gpu() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["bloodsmear"]

    assert "./models:/app/models:ro" in service["volumes"]
    assert "./outputs:/app/outputs" in service["volumes"]
    assert "./data:/app/data" in service["volumes"]
    assert service["gpus"] == "all"
    assert service["user"] == "${BLOODSMEAR_UID:-1000}:${BLOODSMEAR_GID:-1000}"
    assert service["environment"]["BLOODSMEAR_REQUIRE_GPU"] == "true"
    assert service["environment"]["BLOODSMEAR_DATABASE_PATH"] == "/app/data/bloodsmear.db"
    assert service["environment"]["BLOODSMEAR_JOBS_ROOT"] == "/app/data/jobs"
    assert "/health/ready" in " ".join(service["healthcheck"]["test"])


def test_runtime_data_is_ignored_by_git() -> None:
    ignore_rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/" in ignore_rules


def test_runtime_lock_excludes_test_only_dependencies() -> None:
    runtime_lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    dev_lock = (ROOT / "requirements-dev.lock").read_text(encoding="utf-8")

    assert "pytest==" not in runtime_lock
    assert "httpx2==" not in runtime_lock
    assert "-r requirements.lock" in dev_lock
    assert "pytest==9.1.1" in dev_lock
    assert "httpx2==2.13.1" in dev_lock
