""" 
JobStore module for managing job runs in the database.
This module provides the JobStore class which allows creating, 
updating, and querying job runs.

It' usefull for long-running tasks like data ingestion or model training,
to track their progress, status, and results.

It uses psycopg for PostgreSQL interactions, 
but degrades gracefully if psycopg is not installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

# Graceful degradation
# If psycopg is not installed, 
# JobStore will be disabled 
# but the code can still run without errors 
# (logs won't be saved in DB)
try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Json
except Exception:  # pragma: no cover - optional dependency
    psycopg = None
    dict_row = None
    Json = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRun:
    run_id: str
    job_type: str
    source: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    duration_ms: Optional[int]
    progress_pct: Optional[int]
    message: Optional[str]
    records_count: int
    pages_count: int
    errors_count: int
    params_json: Optional[Dict[str, Any]]
    result_json: Optional[Dict[str, Any]]
    error_text: Optional[str]
    records_per_sec: Optional[float]
    updated_at: datetime


class JobStore:
    def __init__(self, dsn: Optional[str]) -> None:
        self._dsn = dsn

    # Actvive or not based on presence of DSN (connect string) and psycopg
    @property
    def enabled(self) -> bool:
        return bool(self._dsn) and psycopg is not None

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("JobStore disabled")
        return psycopg.connect(self._dsn, autocommit=True, row_factory=dict_row)

    def create(self, run_id: str, job_type: str, source: str, params: Dict[str, Any], message: str) -> None:
        if not self.enabled:
            return
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                insert into job_runs (
                  run_id, job_type, source, status, started_at, updated_at,
                  progress_pct, message, params_json
                ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (run_id) do update set
                  job_type = excluded.job_type,
                  source = excluded.source,
                  status = excluded.status,
                  started_at = excluded.started_at,
                  updated_at = excluded.updated_at,
                  progress_pct = excluded.progress_pct,
                  message = excluded.message,
                  params_json = excluded.params_json
                """,
                (
                    run_id,
                    job_type,
                    source,
                    "RUNNING",
                    now,
                    now,
                    0,
                    message,
                    Json(params) if Json else json.dumps(params),
                ),
            )

    def progress(
        self,
        run_id: str,
        progress_pct: Optional[int] = None,
        message: Optional[str] = None,
        records_count: Optional[int] = None,
        pages_count: Optional[int] = None,
        errors_count: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        fields: List[str] = ["updated_at = %s"]
        values: List[Any] = [utc_now()]
        if progress_pct is not None:
            fields.append("progress_pct = %s")
            values.append(progress_pct)
        if message is not None:
            fields.append("message = %s")
            values.append(message)
        if records_count is not None:
            fields.append("records_count = %s")
            values.append(records_count)
        if pages_count is not None:
            fields.append("pages_count = %s")
            values.append(pages_count)
        if errors_count is not None:
            fields.append("errors_count = %s")
            values.append(errors_count)
        if result is not None:
            fields.append("result_json = %s")
            values.append(Json(result) if Json else json.dumps(result))

        values.append(run_id)
        query = "update job_runs set " + ", ".join(fields) + " where run_id = %s"

        with self._connect() as conn:
            conn.execute(query, values)

    def finish(self, run_id: str, status: str, result: Optional[Dict[str, Any]] = None, error_text: Optional[str] = None) -> None:
        if not self.enabled:
            return
        ended_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                update job_runs
                set status = %s,
                    ended_at = %s,
                    duration_ms = (extract(epoch from (%s - started_at)) * 1000)::bigint,
                    records_per_sec = case 
                        when (extract(epoch from (%s - started_at)) * 1000)::bigint > 0 
                        then round((records_count * 1000.0) / nullif((extract(epoch from (%s - started_at)) * 1000)::bigint, 0), 3)
                        else 0
                    end,
                    progress_pct = case when %s = 'SUCCESS' then 100 else progress_pct end,
                    result_json = coalesce(%s::jsonb, result_json),
                    error_text = %s,
                    updated_at = %s
                where run_id = %s
                """,
                (
                    status,
                    ended_at,
                    ended_at,
                    ended_at,
                    ended_at,
                    status,
                    Json(result) if (Json and result is not None) else (json.dumps(result) if result is not None else None),
                    error_text,
                    ended_at,
                    run_id,
                ),
            )

    @staticmethod
    def _normalize_row(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Renomme result_json → result pour exposer une structure cohérente."""
        if row is None:
            return None
        if "result_json" in row:
            row = dict(row)
            row["result"] = row.pop("result_json")
        return row

    def list_runs(self, source: Optional[str], status: Optional[str], limit: int) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        filters: List[str] = []
        values: List[Any] = []
        if source:
            filters.append("source = %s")
            values.append(source)
        if status:
            filters.append("status = %s")
            values.append(status)
        where_clause = f"where {' and '.join(filters)}" if filters else ""
        values.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select * from job_runs
                {where_clause}
                order by started_at desc
                limit %s
                """,
                values,
            ).fetchall()
        return [self._normalize_row(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self._connect() as conn:
            row = conn.execute("select * from job_runs where run_id = %s", (run_id,)).fetchone()
        return self._normalize_row(row)

    def mark_stale(self, stale_minutes: int) -> int:
        if not self.enabled:
            return 0
        cutoff = utc_now() - timedelta(minutes=stale_minutes)
        with self._connect() as conn:
            result = conn.execute(
                """
                update job_runs
                set status = 'FAILED',
                    error_text = 'stale job detected (api restart or crash)',
                    updated_at = %s
                where status = 'RUNNING' and updated_at < %s
                """,
                (utc_now(), cutoff),
            )
            return result.rowcount or 0
