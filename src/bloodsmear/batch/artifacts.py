from __future__ import annotations

import json
import os
import re
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
    # Public ZIP names are independent of private storage paths and item IDs.
    export_directories = {
        item.id: _zip_result_directory(item)
        for item in items
        if item.status == ItemStatus.COMPLETED and item.result_directory
    }
    exported_summary = summary.model_copy(update={
        "items": [entry.model_copy(update={
            "result_directory": export_directories.get(entry.item_id)
        }) for entry in summary.items]
    })
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
                members.append((path, f"{export_directories[item.id]}/{filename}"))
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, archive_name in sorted(members, key=lambda pair: pair[1]):
                if archive_name == "summary.json":
                    archive.writestr(archive_name, exported_summary.model_dump_json(indent=2))
                else:
                    archive.write(path, archive_name)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _zip_result_directory(item: BatchItem) -> str:
    # A single Windows-safe component: never interpret an uploaded name as a path.
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", item.original_filename).rstrip(" .")
    filename = filename or "image"
    if len(filename) > 120 or len(filename.encode("utf-8")) > 240:
        suffix = Path(filename).suffix
        # Accepted image suffixes are short; do not let an arbitrary long suffix defeat the limit.
        suffix = suffix if len(suffix) <= 16 else ""
        stem = filename[:-len(suffix)] if suffix else filename
        stem = stem[:120 - len(suffix)]
        # Linux/macOS limits count encoded bytes, whereas Windows counts UTF-16 units.
        # Decode only the valid prefix so truncation never leaves a partial Unicode character.
        stem = stem.encode("utf-8")[:240 - len(suffix.encode("utf-8"))].decode("utf-8", errors="ignore")
        filename = stem.rstrip(" .") + suffix
    return f"items/{item.ordinal + 1:03d}_{filename}"
