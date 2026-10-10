"""Keep only the latest startup diagnostic, without sample or patient information."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from bloodsmear.config import AppSettings


LOGGER = logging.getLogger("bloodsmear.startup")


def record_startup_status(
    settings: AppSettings, status: str, *, provider: str | None = None,
    error_code: str | None = None,
) -> None:
    temporary: Path | None = None
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
            "port": settings.port, "require_gpu": settings.require_gpu,
            "provider": provider, "error_code": error_code,
        }
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=settings.log_dir,
            prefix=".startup-status-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False)
            stream.write("\n")
        temporary.replace(settings.log_dir / "startup-status.json")
    except OSError:
        # A diagnostic-write error must not hide the original GPU/model failure.
        LOGGER.warning("无法保存启动诊断记录，请检查日志目录权限和磁盘空间。")
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
