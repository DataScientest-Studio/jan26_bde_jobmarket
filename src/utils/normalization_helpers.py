"""Shared helpers for silver normalization and merge pipelines.

This module centralizes low-level, reusable helpers used across source
normalization and merge code paths:
- safe type conversions (float, datetime)
- dataframe persistence with a consistent output format contract
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Optional


def safe_str_to_float(value: Any) -> float:
    """Convert mixed scalar input to float, using 0.0 for invalid values."""
    if value is None or value == "":
        return 0.0
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return 0.0
    return 0.0


def safe_str_to_datetime(value: Any, formats: Optional[Iterable[str]] = None) -> Optional[datetime]:
    """Convert mixed scalar input to datetime, returning None on parse failure."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        date_formats = list(formats) if formats else [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y",
        ]
        for fmt in date_formats:
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
    return None


def write_dataframe_with_format(storage, df, key_base: str, output_format: str) -> str:
    """Write a DataFrame to storage in parquet/jsonl/csv and return target key.

    Args:
        storage: Project storage backend instance.
        df: Pandas DataFrame to persist.
        key_base: Key prefix without extension.
        output_format: One of "parquet", "jsonl", "csv".
    """
    if output_format == "parquet":
        target_key = f"{key_base}.parquet"
        storage.write_parquet(target_key, df)
        return target_key

    if output_format == "jsonl":
        target_key = f"{key_base}.jsonl"
        storage.write_jsonl(target_key, df.to_dict(orient="records"))
        return target_key

    if output_format == "csv":
        target_key = f"{key_base}.csv"
        storage.write_bytes(target_key, df.to_csv(index=False).encode("utf-8"), content_type="text/csv")
        return target_key

    raise ValueError(f"Unsupported output_format: {output_format}")
