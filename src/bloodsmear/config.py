from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BLOODSMEAR_",
        env_file=".env",
        extra="ignore",
    )

    model_dir: Path = Path("models/blood-cell-yolo11/1.0.0")
    output_dir: Path = Path("outputs")
    max_upload_bytes: int = 25 * 1024 * 1024
    require_gpu: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    input_size: int = Field(default=640, gt=0)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_detections: int = Field(default=3000, gt=0)
    database_path: Path = Path("data/bloodsmear.db")
    jobs_root: Path = Path("data/jobs")
    batch_max_files: int = Field(default=100, gt=0)
    batch_max_file_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    batch_max_total_bytes: int = Field(default=500 * 1024 * 1024, gt=0)
    input_retention_hours: int = Field(default=24, gt=0)
    result_retention_days: int = Field(default=7, gt=0)
    cleanup_interval_seconds: int = Field(default=3600, gt=0)
    batch_poll_interval_seconds: float = Field(default=0.5, gt=0)
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    log_retention_days: int = Field(default=14, gt=0)
    log_to_console: bool = True
