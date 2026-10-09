from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from bloodsmear.batch.aggregation import BatchSummary, build_batch_summary
from bloodsmear.batch.domain import BatchItem, BatchJob, ItemStatus


@dataclass(frozen=True)
class BatchArtifactPaths:
    summary_path: Path
    failures_path: Path
    zip_path: Path


def write_batch_artifacts(
    job: BatchJob,
    items: Sequence[BatchItem],
    job_root: Path,
    classes: Sequence[str],
) -> BatchArtifactPaths:
    job_root.mkdir(parents=True, exist_ok=True)
    summary = build_batch_summary(job, items, classes)
    summary_path = _atomic_text(
        job_root / "summary.json",
        summary.model_dump_json(indent=2),
    )
    failures_path = _atomic_text(
        job_root / "failures.json",
        json.dumps(summary.failures, ensure_ascii=False, indent=2),
    )
    zip_path = _write_zip(summary, items, job_root)
    return BatchArtifactPaths(summary_path, failures_path, zip_path)


def _atomic_text(destination: Path, content: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _write_zip(
    summary: BatchSummary,
    items: Sequence[BatchItem],
    job_root: Path,
) -> Path:
    destination = job_root / "results.zip"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".results.", suffix=".tmp", dir=job_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    members: list[tuple[Path, str]] = [
        (job_root / "summary.json", "summary.json"),
        (job_root / "failures.json", "failures.json"),
    ]
    for report_name in ("report.xlsx", "report.pdf"):
        report_path = job_root / report_name
        if report_path.is_file():
            members.append((report_path, report_name))
    for item in sorted(items, key=lambda entry: entry.ordinal):
        if item.status != ItemStatus.COMPLETED or not item.result_directory:
            continue
        item_root = job_root / item.result_directory
        for filename in (
            "result.json",
            "detections.csv",
            "annotated.png",
            "report.xlsx",
            "report.pdf",
        ):
            path = item_root / filename
            if path.is_file():
                members.append((path, f"items/{item.id}/{filename}"))
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, archive_name in sorted(members, key=lambda pair: pair[1]):
                archive.write(path, archive_name)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
