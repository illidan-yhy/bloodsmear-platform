from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bloodsmear.batch.aggregation import build_batch_summary
from bloodsmear.batch.domain import (
    BatchItem,
    BatchJob,
    BatchMode,
    ItemStatus,
    JobStatus,
)
from bloodsmear.domain import ResultSummary


CLASSES = [
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelets",
    "RBC",
]
NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def result_summary(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    wbc_total = sum(counts[name] for name in CLASSES[:5])
    summary = ResultSummary(
        total_detected_cells=total,
        cell_counts=counts,
        all_cell_ratios={name: counts[name] / total if total else None for name in CLASSES},
        wbc_differential_ratios={
            name: counts[name] / wbc_total if wbc_total else None for name in CLASSES[:5]
        },
    )
    return summary.model_dump_json()


def job(mode: BatchMode, status: JobStatus, total: int = 2) -> BatchJob:
    return BatchJob(
        id="job",
        mode=mode,
        sample_id="S001" if mode == BatchMode.GROUPED else None,
        status=status,
        total_items=total,
        completed_items=0,
        failed_items=0,
        created_at=NOW,
        input_expires_at=NOW + timedelta(hours=24),
        result_expires_at=NOW + timedelta(days=7),
    )


def item(
    ordinal: int,
    status: ItemStatus,
    *,
    summary: str | None = None,
) -> BatchItem:
    return BatchItem(
        id=f"item-{ordinal}",
        job_id="job",
        ordinal=ordinal,
        original_filename=f"field-{ordinal}.png",
        stored_filename=f"item-{ordinal}.png",
        input_path=f"inputs/item-{ordinal}.png",
        sample_id="S001",
        view_id=f"view-{ordinal}",
        status=status,
        result_directory=f"items/item-{ordinal}" if status == ItemStatus.COMPLETED else None,
        result_summary_json=summary,
        error_code="ITEM_INFERENCE_FAILED" if status == ItemStatus.FAILED else None,
        error_message="Inference failed" if status == ItemStatus.FAILED else None,
    )


def test_independent_summary_has_no_cross_sample_aggregate_and_orders_items() -> None:
    counts = {name: int(name == "RBC") for name in CLASSES}
    items = [item(1, ItemStatus.COMPLETED, summary=result_summary(counts)), item(0, ItemStatus.COMPLETED, summary=result_summary(counts))]

    summary = build_batch_summary(job(BatchMode.INDEPENDENT, JobStatus.COMPLETED), items, CLASSES)

    assert summary.aggregate_summary is None
    assert [entry.ordinal for entry in summary.items] == [0, 1]


def test_grouped_summary_sums_counts_then_recomputes_ratios() -> None:
    first = {name: 0 for name in CLASSES}
    first.update({"Neutrophil": 3, "Lymphocyte": 1, "RBC": 10})
    second = {name: 0 for name in CLASSES}
    second.update({"Neutrophil": 1, "Lymphocyte": 3, "RBC": 20})

    summary = build_batch_summary(
        job(BatchMode.GROUPED, JobStatus.COMPLETED),
        [
            item(0, ItemStatus.COMPLETED, summary=result_summary(first)),
            item(1, ItemStatus.COMPLETED, summary=result_summary(second)),
        ],
        CLASSES,
    )

    assert summary.aggregate_summary is not None
    assert summary.aggregate_summary.cell_counts["RBC"] == 30
    assert summary.aggregate_summary.cell_counts["Neutrophil"] == 4
    assert summary.aggregate_summary.wbc_differential_ratios["Neutrophil"] == 0.5


def test_grouped_partial_failure_adds_warning_and_sanitized_failure() -> None:
    counts = {name: int(name == "RBC") for name in CLASSES}

    summary = build_batch_summary(
        job(BatchMode.GROUPED, JobStatus.PARTIAL_FAILED),
        [
            item(0, ItemStatus.COMPLETED, summary=result_summary(counts)),
            item(1, ItemStatus.FAILED),
        ],
        CLASSES,
    )

    assert "INCOMPLETE_GROUP" in summary.warnings
    assert summary.failures == [
        {
            "item_id": "item-1",
            "original_filename": "field-1.png",
            "error_code": "ITEM_INFERENCE_FAILED",
            "error_message": "Inference failed",
        }
    ]


def test_grouped_zero_success_has_null_ratios() -> None:
    summary = build_batch_summary(
        job(BatchMode.GROUPED, JobStatus.FAILED),
        [item(0, ItemStatus.FAILED), item(1, ItemStatus.FAILED)],
        CLASSES,
    )

    assert summary.aggregate_summary is not None
    assert summary.aggregate_summary.total_detected_cells == 0
    assert all(value is None for value in summary.aggregate_summary.all_cell_ratios.values())
    assert all(
        value is None
        for value in summary.aggregate_summary.wbc_differential_ratios.values()
    )
