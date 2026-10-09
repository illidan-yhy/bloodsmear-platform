from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from bloodsmear.batch.domain import BatchItem, BatchJob, BatchMode, ItemStatus
from bloodsmear.domain import ResultSummary, WBC_CLASSES


class BatchItemIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    ordinal: int
    original_filename: str
    sample_id: str
    view_id: str
    status: ItemStatus
    result_directory: str | None = None
    summary: ResultSummary | None = None
    error_code: str | None = None
    error_message: str | None = None


class BatchSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    mode: BatchMode
    status: str
    sample_id: str | None
    total_views: int
    successful_views: int
    failed_views: int
    items: list[BatchItemIndex]
    aggregate_summary: ResultSummary | None
    warnings: list[str]
    failures: list[dict[str, str]]


def build_batch_summary(
    job: BatchJob,
    items: Sequence[BatchItem],
    classes: Sequence[str],
) -> BatchSummary:
    ordered = sorted(items, key=lambda item: item.ordinal)
    indexes: list[BatchItemIndex] = []
    failures: list[dict[str, str]] = []
    successful_summaries: list[ResultSummary] = []
    for item in ordered:
        parsed_summary = (
            ResultSummary.model_validate_json(item.result_summary_json)
            if item.result_summary_json
            else None
        )
        if parsed_summary is not None:
            successful_summaries.append(parsed_summary)
        if item.status == ItemStatus.FAILED:
            failures.append(
                {
                    "item_id": item.id,
                    "original_filename": item.original_filename,
                    "error_code": item.error_code or "ITEM_INFERENCE_FAILED",
                    "error_message": item.error_message or "Inference failed",
                }
            )
        indexes.append(
            BatchItemIndex(
                item_id=item.id,
                ordinal=item.ordinal,
                original_filename=item.original_filename,
                sample_id=item.sample_id,
                view_id=item.view_id,
                status=item.status,
                result_directory=item.result_directory,
                summary=parsed_summary,
                error_code=item.error_code,
                error_message=item.error_message,
            )
        )

    warnings: list[str] = []
    aggregate: ResultSummary | None = None
    if job.mode == BatchMode.GROUPED:
        counts = {name: 0 for name in classes}
        for summary in successful_summaries:
            for name in classes:
                counts[name] += summary.cell_counts[name]
        aggregate, aggregate_warnings = _summary_from_counts(counts, classes)
        warnings.extend(aggregate_warnings)
        if failures:
            warnings.insert(0, "INCOMPLETE_GROUP")

    return BatchSummary(
        job_id=job.id,
        mode=job.mode,
        status=job.status.value,
        sample_id=job.sample_id,
        total_views=len(ordered),
        successful_views=len(successful_summaries),
        failed_views=len(failures),
        items=indexes,
        aggregate_summary=aggregate,
        warnings=list(dict.fromkeys(warnings)),
        failures=failures,
    )


def _summary_from_counts(
    counts: dict[str, int],
    classes: Sequence[str],
) -> tuple[ResultSummary, list[str]]:
    total = sum(counts.values())
    all_ratios = {
        name: counts[name] / total if total else None
        for name in classes
    }
    wbc_names = [name for name in classes if name in WBC_CLASSES]
    wbc_total = sum(counts[name] for name in wbc_names)
    wbc_ratios = {
        name: counts[name] / wbc_total if wbc_total else None
        for name in wbc_names
    }
    warnings = [] if wbc_total else ["NO_WBC_DETECTED"]
    return (
        ResultSummary(
            total_detected_cells=total,
            cell_counts=counts,
            all_cell_ratios=all_ratios,
            wbc_differential_ratios=wbc_ratios,
        ),
        warnings,
    )
