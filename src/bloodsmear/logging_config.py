from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from bloodsmear.config import AppSettings


FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(settings: AppSettings) -> Path:
    log_dir = settings.log_dir
    signature = (
        str(log_dir.resolve()),
        settings.log_level.upper(),
        settings.log_retention_days,
        settings.log_to_console,
    )
    root = logging.getLogger()
    if getattr(root, "_bloodsmear_logging_signature", None) == signature:
        return log_dir / "app.log"
    reset_managed_logging()
    log_dir.mkdir(parents=True, exist_ok=True)
    cleanup_expired_logs(log_dir, settings.log_retention_days)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(FORMAT, datefmt="%Y-%m-%dT%H:%M:%S%z")
    file_handler = TimedRotatingFileHandler(
        log_dir / "app.log",
        when="midnight",
        interval=1,
        backupCount=settings.log_retention_days,
        encoding="utf-8",
        delay=True,
        utc=False,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    file_handler._bloodsmear_managed = True
    root.addHandler(file_handler)
    if settings.log_to_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(level)
        console._bloodsmear_managed = True
        root.addHandler(console)
    root.setLevel(level)
    root._bloodsmear_logging_signature = signature
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).propagate = True
    return log_dir / "app.log"


def cleanup_expired_logs(
    log_dir: Path,
    retention_days: int,
    now: datetime | None = None,
) -> list[Path]:
    current = now or datetime.now(timezone.utc)
    cutoff = current - timedelta(days=retention_days)
    removed: list[Path] = []
    if not log_dir.exists():
        return removed
    for path in log_dir.glob("app.log.*"):
        if not path.is_file():
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified < cutoff:
            path.unlink()
            removed.append(path)
    return removed


def reset_managed_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_bloodsmear_managed", False):
            root.removeHandler(handler)
            handler.close()
    if hasattr(root, "_bloodsmear_logging_signature"):
        delattr(root, "_bloodsmear_logging_signature")
