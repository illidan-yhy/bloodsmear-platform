from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
from uuid import uuid4

from bloodsmear.config import AppSettings
from bloodsmear.errors import (
    BatchTotalSizeExceededError,
    InvalidImageError,
    UnsafeJobPathError,
)
from bloodsmear.image_ops import SUPPORTED_EXTENSIONS


CHUNK_SIZE = 1024 * 1024


class AsyncUpload(Protocol):
    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True)
class StagedUpload:
    item_id: str
    ordinal: int
    original_filename: str
    stored_filename: str
    input_path: str
    size_bytes: int


class BatchStorage:
    def __init__(self, settings: AppSettings) -> None:
        self.jobs_root = settings.jobs_root
        self.staging_root = settings.jobs_root / ".staging"
        self.max_file_bytes = settings.batch_max_file_bytes
        self.max_total_bytes = settings.batch_max_total_bytes

    async def stage_uploads(
        self,
        job_id: str,
        files: Sequence[AsyncUpload],
    ) -> list[StagedUpload]:
        normalized: list[tuple[str, str]] = []
        for upload in files:
            original = Path(upload.filename or "").name
            extension = Path(original).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                raise InvalidImageError(
                    f"Unsupported image extension: {extension or '<none>'}"
                )
            normalized.append((original, extension))

        staging_job = self.staging_root / job_id
        inputs_dir = staging_job / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=False)
        staged: list[StagedUpload] = []
        total_bytes = 0
        try:
            for ordinal, (upload, name_info) in enumerate(zip(files, normalized)):
                original, extension = name_info
                item_id = uuid4().hex
                stored_filename = f"{item_id}{extension}"
                destination = inputs_dir / stored_filename
                file_bytes = 0
                with destination.open("wb") as handle:
                    while True:
                        chunk = await upload.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        file_bytes += len(chunk)
                        total_bytes += len(chunk)
                        if file_bytes > self.max_file_bytes:
                            raise InvalidImageError(
                                "Image exceeds the per-file upload limit"
                            )
                        if total_bytes > self.max_total_bytes:
                            raise BatchTotalSizeExceededError(
                                "Batch exceeds the total upload limit"
                            )
                        handle.write(chunk)
                staged.append(
                    StagedUpload(
                        item_id=item_id,
                        ordinal=ordinal,
                        original_filename=original,
                        stored_filename=stored_filename,
                        input_path=f"inputs/{stored_filename}",
                        size_bytes=file_bytes,
                    )
                )
            return staged
        except Exception:
            self.discard_staging(job_id)
            raise

    def promote(self, job_id: str) -> Path:
        source = self.staging_root / job_id
        destination = self.job_root(job_id)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        self._remove_staging_root_if_empty()
        return destination

    def discard_staging(self, job_id: str) -> None:
        target = self.staging_root / job_id
        if target.exists():
            shutil.rmtree(target)
        self._remove_staging_root_if_empty()

    def _remove_staging_root_if_empty(self) -> None:
        if self.staging_root.exists() and not any(self.staging_root.iterdir()):
            self.staging_root.rmdir()

    def job_root(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def guarded_remove(self, job_id: str, relative_path: str | None = None) -> None:
        if ".." in Path(job_id).parts:
            raise UnsafeJobPathError("Refusing path traversal")
        if relative_path and ".." in Path(relative_path).parts:
            raise UnsafeJobPathError("Refusing path traversal")
        root = self.jobs_root.resolve()
        candidate = self.job_root(job_id)
        if relative_path:
            candidate = candidate / relative_path
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root) or candidate.is_symlink():
            raise UnsafeJobPathError("Refusing to delete a path outside jobs_root")
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(character if character.isalnum() else "-" for character in stem)
    return cleaned.strip("-")[:64] or "sample"
