from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from bloodsmear.config import AppSettings
from bloodsmear.logging_config import cleanup_expired_logs, configure_logging, reset_managed_logging


def teardown_function() -> None:
    reset_managed_logging()


def test_logging_settings_defaults_and_environment_override(monkeypatch) -> None:
    defaults = AppSettings()
    assert defaults.log_dir == Path("logs")
    assert defaults.log_level == "INFO"
    assert defaults.log_retention_days == 14
    assert defaults.log_to_console is True

    monkeypatch.setenv("BLOODSMEAR_LOG_RETENTION_DAYS", "30")
    monkeypatch.setenv("BLOODSMEAR_LOG_TO_CONSOLE", "false")
    configured = AppSettings()
    assert configured.log_retention_days == 30
    assert configured.log_to_console is False


def test_configure_logging_creates_daily_handler_and_is_idempotent(tmp_path: Path) -> None:
    settings = AppSettings(log_dir=tmp_path, log_to_console=False)

    configure_logging(settings)
    configure_logging(settings)

    handlers = [h for h in logging.getLogger().handlers if getattr(h, "_bloodsmear_managed", False)]
    assert len(handlers) == 1
    handler = handlers[0]
    assert isinstance(handler, TimedRotatingFileHandler)
    assert handler.when == "MIDNIGHT"
    assert handler.backupCount == 14
    logging.getLogger("bloodsmear.test").info("hello")
    handler.flush()
    assert "hello" in (tmp_path / "app.log").read_text(encoding="utf-8")


def test_cleanup_removes_only_expired_rotated_logs(tmp_path: Path) -> None:
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    old = tmp_path / "app.log.2026-09-01"
    recent = tmp_path / "app.log.2026-10-06"
    active = tmp_path / "app.log"
    unrelated = tmp_path / "notes.txt"
    for path in (old, recent, active, unrelated):
        path.write_text(path.name, encoding="utf-8")
    old_timestamp = (now - timedelta(days=30)).timestamp()
    recent_timestamp = (now - timedelta(days=1)).timestamp()
    os.utime(old, (old_timestamp, old_timestamp))
    os.utime(recent, (recent_timestamp, recent_timestamp))

    removed = cleanup_expired_logs(tmp_path, retention_days=14, now=now)

    assert removed == [old]
    assert not old.exists()
    assert recent.exists()
    assert active.exists()
    assert unrelated.exists()
