from __future__ import annotations

import sqlite3
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from bloodsmear.batch.domain import (
    BatchItem,
    BatchJob,
    BatchJobDetail,
    BatchMode,
    ItemStatus,
    JobStatus,
    TERMINAL_JOB_STATUSES,
)
from bloodsmear.errors import JobNotFoundError


class BatchRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.database_path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    sample_id TEXT,
                    metadata_json TEXT,
                    status TEXT NOT NULL,
                    total_items INTEGER NOT NULL,
                    completed_items INTEGER NOT NULL DEFAULT 0,
                    failed_items INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    input_expires_at TEXT NOT NULL,
                    inputs_deleted_at TEXT,
                    result_expires_at TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS batch_items (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES batch_jobs(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    view_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_directory TEXT,
                    result_summary_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    inference_ms REAL,
                    UNIQUE(job_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_batch_jobs_status_created
                    ON batch_jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_batch_jobs_input_expiry
                    ON batch_jobs(input_expires_at);
                CREATE INDEX IF NOT EXISTS idx_batch_jobs_result_expiry
                    ON batch_jobs(result_expires_at);
                CREATE INDEX IF NOT EXISTS idx_batch_items_job_ordinal
                    ON batch_items(job_id, ordinal);
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(batch_jobs)")}
            if "metadata_json" not in columns:
                connection.execute("ALTER TABLE batch_jobs ADD COLUMN metadata_json TEXT")

    def create_job(
        self,
        job: BatchJob,
        items: Sequence[BatchItem],
        promote: Callable[[], None],
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO batch_jobs (
                    id, mode, sample_id, metadata_json, status, total_items, completed_items,
                    failed_items, created_at, started_at, completed_at,
                    input_expires_at, result_expires_at, error_code, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _job_values(job),
            )
            connection.executemany(
                """
                INSERT INTO batch_items (
                    id, job_id, ordinal, original_filename, stored_filename,
                    input_path, sample_id, view_id, status, result_directory,
                    result_summary_json, error_code, error_message, inference_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_item_values(item) for item in items],
            )
            promote()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim_next_job(self, now: datetime) -> BatchJob | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM batch_jobs
                WHERE status = ?
                ORDER BY created_at, id
                LIMIT 1
                """,
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                "UPDATE batch_jobs SET status = ?, started_at = ? WHERE id = ? AND status = ?",
                (
                    JobStatus.RUNNING.value,
                    _format_datetime(now),
                    row["id"],
                    JobStatus.QUEUED.value,
                ),
            )
            connection.commit()
            claimed = dict(row)
            claimed["status"] = JobStatus.RUNNING.value
            claimed["started_at"] = _format_datetime(now)
            return _job_from_row(claimed)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_job(self, job_id: str) -> BatchJobDetail:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM batch_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        job = _job_from_row(row)
        return BatchJobDetail(**job.model_dump(), items=self.get_items(job_id))

    def get_items(self, job_id: str) -> list[BatchItem]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM batch_items WHERE job_id = ? ORDER BY ordinal",
                (job_id,),
            ).fetchall()
        return [_item_from_row(row) for row in rows]

    def mark_item_running(self, item_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE batch_items SET status = ? WHERE id = ? AND status = ?",
                (ItemStatus.RUNNING.value, item_id, ItemStatus.PENDING.value),
            )

    def complete_item(
        self,
        item_id: str,
        result_directory: str,
        result_summary_json: str,
        inference_ms: float,
    ) -> None:
        self._finish_item(
            item_id,
            status=ItemStatus.COMPLETED,
            result_directory=result_directory,
            result_summary_json=result_summary_json,
            inference_ms=inference_ms,
        )

    def fail_item(self, item_id: str, error_code: str, error_message: str) -> None:
        self._finish_item(
            item_id,
            status=ItemStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
        )

    def _finish_item(
        self,
        item_id: str,
        *,
        status: ItemStatus,
        result_directory: str | None = None,
        result_summary_json: str | None = None,
        inference_ms: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT job_id FROM batch_items WHERE id = ?", (item_id,)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Item not found: {item_id}")
            connection.execute(
                """
                UPDATE batch_items
                SET status = ?, result_directory = ?, result_summary_json = ?,
                    inference_ms = ?, error_code = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    result_directory,
                    result_summary_json,
                    inference_ms,
                    error_code,
                    error_message,
                    item_id,
                ),
            )
            self._recompute_progress(connection, row["job_id"])
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _recompute_progress(self, connection: sqlite3.Connection, job_id: str) -> None:
        connection.execute(
            """
            UPDATE batch_jobs
            SET completed_items = (
                    SELECT COUNT(*) FROM batch_items
                    WHERE job_id = ? AND status = ?
                ),
                failed_items = (
                    SELECT COUNT(*) FROM batch_items
                    WHERE job_id = ? AND status = ?
                )
            WHERE id = ?
            """,
            (
                job_id,
                ItemStatus.COMPLETED.value,
                job_id,
                ItemStatus.FAILED.value,
                job_id,
            ),
        )

    def finalize_job(
        self,
        job_id: str,
        status: JobStatus,
        completed_at: datetime,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if status not in TERMINAL_JOB_STATUSES:
            raise ValueError("final job status must be terminal")
        with self._connect() as connection:
            updated = connection.execute(
                "UPDATE batch_jobs SET status = ?, completed_at = ?, error_code = ?, error_message = ? WHERE id = ?",
                (status.value, _format_datetime(completed_at), error_code, error_message, job_id),
            ).rowcount
        if not updated:
            raise JobNotFoundError(f"Job not found: {job_id}")

    def recover_interrupted(self) -> int:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            job_ids = [
                row["id"]
                for row in connection.execute(
                    "SELECT id FROM batch_jobs WHERE status = ?",
                    (JobStatus.RUNNING.value,),
                ).fetchall()
            ]
            for job_id in job_ids:
                connection.execute(
                    "UPDATE batch_items SET status = ? WHERE job_id = ? AND status = ?",
                    (ItemStatus.PENDING.value, job_id, ItemStatus.RUNNING.value),
                )
                connection.execute(
                    "UPDATE batch_jobs SET status = ?, started_at = NULL WHERE id = ?",
                    (JobStatus.QUEUED.value, job_id),
                )
            connection.commit()
            return len(job_ids)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def delete_job(self, job_id: str) -> None:
        with self._connect() as connection:
            deleted = connection.execute(
                "DELETE FROM batch_jobs WHERE id = ?", (job_id,)
            ).rowcount
        if not deleted:
            raise JobNotFoundError(f"Job not found: {job_id}")

    def list_expired_inputs(self, now: datetime) -> list[BatchJob]:
        return self._list_expired("input_expires_at", now, inputs_only=True)

    def list_expired_results(self, now: datetime) -> list[BatchJob]:
        return self._list_expired("result_expires_at", now, inputs_only=False)

    def _list_expired(
        self,
        column: str,
        now: datetime,
        *,
        inputs_only: bool,
    ) -> list[BatchJob]:
        terminal_values = tuple(status.value for status in TERMINAL_JOB_STATUSES)
        placeholders = ",".join("?" for _ in terminal_values)
        input_filter = "AND inputs_deleted_at IS NULL" if inputs_only else ""
        query = f"""
            SELECT * FROM batch_jobs
            WHERE status IN ({placeholders})
              AND {column} <= ?
              {input_filter}
            ORDER BY {column}, id
        """
        with self._connect() as connection:
            rows = connection.execute(
                query,
                (*terminal_values, _format_datetime(now)),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def mark_inputs_removed(self, job_id: str, removed_at: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE batch_jobs SET inputs_deleted_at = ? WHERE id = ?",
                (_format_datetime(removed_at), job_id),
            )

    def count_items(self, job_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM batch_items WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return int(row["count"])


def _format_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _job_values(job: BatchJob) -> tuple[object, ...]:
    return (
        job.id,
        job.mode.value,
        job.sample_id,
        job.metadata.model_dump_json() if job.metadata else None,
        job.status.value,
        job.total_items,
        job.completed_items,
        job.failed_items,
        _format_datetime(job.created_at),
        _format_datetime(job.started_at),
        _format_datetime(job.completed_at),
        _format_datetime(job.input_expires_at),
        _format_datetime(job.result_expires_at),
        job.error_code,
        job.error_message,
    )


def _item_values(item: BatchItem) -> tuple[object, ...]:
    return (
        item.id,
        item.job_id,
        item.ordinal,
        item.original_filename,
        item.stored_filename,
        item.input_path,
        item.sample_id,
        item.view_id,
        item.status.value,
        item.result_directory,
        item.result_summary_json,
        item.error_code,
        item.error_message,
        item.inference_ms,
    )


def _job_from_row(row: sqlite3.Row | dict[str, object]) -> BatchJob:
    return BatchJob(
        id=row["id"],
        mode=BatchMode(row["mode"]),
        sample_id=row["sample_id"],
        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else None,
        status=JobStatus(row["status"]),
        total_items=row["total_items"],
        completed_items=row["completed_items"],
        failed_items=row["failed_items"],
        created_at=_parse_datetime(row["created_at"]),
        started_at=_parse_datetime(row["started_at"]),
        completed_at=_parse_datetime(row["completed_at"]),
        input_expires_at=_parse_datetime(row["input_expires_at"]),
        result_expires_at=_parse_datetime(row["result_expires_at"]),
        error_code=row["error_code"],
        error_message=row["error_message"],
    )


def _item_from_row(row: sqlite3.Row) -> BatchItem:
    return BatchItem(
        id=row["id"],
        job_id=row["job_id"],
        ordinal=row["ordinal"],
        original_filename=row["original_filename"],
        stored_filename=row["stored_filename"],
        input_path=row["input_path"],
        sample_id=row["sample_id"],
        view_id=row["view_id"],
        status=ItemStatus(row["status"]),
        result_directory=row["result_directory"],
        result_summary_json=row["result_summary_json"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        inference_ms=row["inference_ms"],
    )
