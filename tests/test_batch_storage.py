from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bloodsmear.batch.domain import BatchMode
from bloodsmear.batch.manager import BatchManager
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage
from bloodsmear.config import AppSettings
from bloodsmear.errors import (
    BatchTooManyFilesError,
    BatchTotalSizeExceededError,
    GroupSampleIdRequiredError,
    InvalidImageError,
    JobNotFoundError,
)


class FakeUpload:
    def __init__(self, filename: str, data: bytes) -> None:
        self.filename = filename
        self.data = data
        self.position = 0
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            size = len(self.data) - self.position
        chunk = self.data[self.position : self.position + size]
        self.position += len(chunk)
        return chunk


def build_manager(
    tmp_path: Path,
    *,
    max_files: int = 100,
    max_file_bytes: int = 25 * 1024 * 1024,
    max_total_bytes: int = 500 * 1024 * 1024,
) -> tuple[BatchManager, BatchRepository, BatchStorage]:
    settings = AppSettings(
        database_path=tmp_path / "batch.db",
        jobs_root=tmp_path / "jobs",
        batch_max_files=max_files,
        batch_max_file_bytes=max_file_bytes,
        batch_max_total_bytes=max_total_bytes,
    )
    repository = BatchRepository(settings.database_path)
    repository.initialize()
    storage = BatchStorage(settings)
    ids = deque(f"{number:032x}" for number in range(1000))
    manager = BatchManager(
        repository,
        storage,
        settings,
        clock=lambda: datetime(2026, 10, 3, tzinfo=timezone.utc),
        id_factory=ids.popleft,
    )
    return manager, repository, storage


@pytest.mark.asyncio
async def test_accepts_100_files_and_rejects_101(tmp_path: Path) -> None:
    manager, _, _ = build_manager(tmp_path)

    accepted = await manager.create_job(
        BatchMode.INDEPENDENT,
        None,
        [FakeUpload(f"{index}.png", b"x") for index in range(100)],
    )

    assert accepted.total_items == 100
    with pytest.raises(BatchTooManyFilesError):
        await manager.create_job(
            BatchMode.INDEPENDENT,
            None,
            [FakeUpload(f"x-{index}.png", b"x") for index in range(101)],
        )


@pytest.mark.asyncio
async def test_file_and_batch_size_boundaries_are_streamed(tmp_path: Path) -> None:
    manager, _, _ = build_manager(
        tmp_path,
        max_file_bytes=3 * 1024 * 1024,
        max_total_bytes=4 * 1024 * 1024,
    )
    large = FakeUpload("large.png", b"x" * (3 * 1024 * 1024))

    await manager.create_job(BatchMode.INDEPENDENT, None, [large])

    assert max(large.read_sizes) <= 1024 * 1024

    manager, _, _ = build_manager(
        tmp_path / "single-over",
        max_file_bytes=5,
        max_total_bytes=10,
    )
    with pytest.raises(InvalidImageError, match="per-file"):
        await manager.create_job(
            BatchMode.INDEPENDENT, None, [FakeUpload("large.png", b"123456")]
        )

    manager, _, _ = build_manager(
        tmp_path / "batch-over",
        max_file_bytes=10,
        max_total_bytes=10,
    )
    with pytest.raises(BatchTotalSizeExceededError):
        await manager.create_job(
            BatchMode.INDEPENDENT,
            None,
            [FakeUpload("a.png", b"123456"), FakeUpload("b.png", b"12345")],
        )


@pytest.mark.asyncio
async def test_duplicate_names_use_unique_paths_and_independent_sample_ids(
    tmp_path: Path,
) -> None:
    manager, _, _ = build_manager(tmp_path)

    detail = await manager.create_job(
        BatchMode.INDEPENDENT,
        None,
        [FakeUpload("same.png", b"a"), FakeUpload("same.png", b"b")],
    )

    assert detail.items[0].stored_filename != detail.items[1].stored_filename
    assert detail.items[0].input_path != detail.items[1].input_path
    assert detail.items[0].sample_id != detail.items[1].sample_id
    assert all((tmp_path / "jobs" / detail.id / item.input_path).is_file() for item in detail.items)


@pytest.mark.asyncio
async def test_grouped_requires_sample_id_before_writing(tmp_path: Path) -> None:
    manager, _, storage = build_manager(tmp_path)

    with pytest.raises(GroupSampleIdRequiredError):
        await manager.create_job(
            BatchMode.GROUPED, "  ", [FakeUpload("one.png", b"x")]
        )

    assert not storage.staging_root.exists()


@pytest.mark.asyncio
async def test_unsupported_extension_creates_no_database_job(tmp_path: Path) -> None:
    manager, repository, storage = build_manager(tmp_path)

    with pytest.raises(InvalidImageError, match="extension"):
        await manager.create_job(
            BatchMode.INDEPENDENT, None, [FakeUpload("one.bmp", b"x")]
        )

    assert not storage.staging_root.exists()
    with pytest.raises(JobNotFoundError):
        repository.get_job("00000000000000000000000000000000")


@pytest.mark.asyncio
async def test_promotion_failure_rolls_back_rows_and_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, repository, storage = build_manager(tmp_path)

    def fail_promotion(job_id: str) -> Path:
        raise RuntimeError("promotion failed")

    monkeypatch.setattr(storage, "promote", fail_promotion)
    with pytest.raises(RuntimeError, match="promotion failed"):
        await manager.create_job(
            BatchMode.INDEPENDENT, None, [FakeUpload("one.png", b"x")]
        )

    with pytest.raises(JobNotFoundError):
        repository.get_job("00000000000000000000000000000000")
    assert not storage.staging_root.exists()
    assert not storage.job_root("00000000000000000000000000000000").exists()
