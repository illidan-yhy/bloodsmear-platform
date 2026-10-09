from __future__ import annotations

import argparse
import asyncio
import json
import zipfile
from io import BytesIO
from pathlib import Path

from starlette.datastructures import UploadFile

from bloodsmear.batch.domain import BatchMode, JobStatus
from bloodsmear.batch.manager import BatchManager
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage
from bloodsmear.batch.worker import BatchWorker
from bloodsmear.config import AppSettings
from bloodsmear.inference import InferenceService


async def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(
        database_path=output / "batch.db",
        jobs_root=output / "jobs",
    )
    repository = BatchRepository(settings.database_path)
    repository.initialize()
    storage = BatchStorage(settings)
    manager = BatchManager(repository, storage, settings)
    service = InferenceService.from_settings(settings)
    data = Path("samples/Blood.png").read_bytes()
    uploads = [
        UploadFile(BytesIO(data), filename="view-1.png"),
        UploadFile(BytesIO(data), filename="view-2.png"),
    ]
    created = await manager.create_job(BatchMode.GROUPED, "S001", uploads)
    worker = BatchWorker(
        repository,
        storage,
        service,
        service.package.manifest.classes,
    )
    assert worker.run_once() is True
    completed = repository.get_job(created.id)
    assert completed.status == JobStatus.COMPLETED
    assert completed.completed_items == 2
    summary_path = storage.job_root(created.id) / "summary.json"
    archive_path = storage.job_root(created.id) / "results.zip"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = summary["aggregate_summary"]["cell_counts"]
    assert summary["aggregate_summary"]["total_detected_cells"] == 110
    assert counts["RBC"] == 104
    assert counts["Monocyte"] == 4
    assert counts["Platelets"] == 2
    item_results = [
        json.loads(
            (storage.job_root(created.id) / item.result_directory / "result.json").read_text(
                encoding="utf-8"
            )
        )
        for item in completed.items
    ]
    assert all(
        result["runtime"]["provider"] == "CUDAExecutionProvider"
        for result in item_results
    )
    with zipfile.ZipFile(archive_path) as archive:
        assert "summary.json" in archive.namelist()
        assert len([name for name in archive.namelist() if name.endswith("result.json")]) == 2
    return {
        "job_id": created.id,
        "provider": "CUDAExecutionProvider",
        "counts": counts,
        "zip": str(archive_path),
        "inference_ms": [result["runtime"]["inference_ms"] for result in item_results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run grouped batch CUDA smoke test")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = asyncio.run(run(arguments.output))
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
