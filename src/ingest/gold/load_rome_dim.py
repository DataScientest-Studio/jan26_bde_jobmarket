"""ROME dimension loader: builds gold.dim_code_rome from France Travail JSONL.

Source — storage layer bronze/france_travail (S3/MinIO: bronze/france_travail/rome/):
  - rome_metiers.jsonl
    Each line: {"code": "A1101", "libelle": "Conducteur / Conductrice de tracteur enjambeur"}

File is read via get_storage_from_env("bronze", "france_travail") using read_bytes() + BytesIO,
so the same code works against both local filesystem and MinIO/S3.

Grain: one row per rome_code (1 584 codes ROME v4).

The UNKNOWN sentinel row is inserted by the schema DDL (ON CONFLICT DO NOTHING).

Usage:
    python -m src.ingest.gold.load_rome_dim
"""

from __future__ import annotations

import io
import json
import logging
import os
import re

import pandas as pd

from src.config.env import load_project_env
from src.storage.storage import get_storage_from_env

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None

logger = logging.getLogger(__name__)

_FT_STORAGE_LAYER = "bronze"
_FT_STORAGE_SOURCE = "france_travail"
_JSONL_KEY = "rome/rome_metiers.jsonl"


def _build_rome_frame() -> pd.DataFrame:
    """Build a dim_code_rome DataFrame from the ROME JSONL file."""
    storage = get_storage_from_env(_FT_STORAGE_LAYER, _FT_STORAGE_SOURCE)
    raw = storage.read_bytes(_JSONL_KEY).decode("utf-8")

    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        records.append({
            "rome_code": obj["code"].strip(),
            "rome_label": obj.get("libelle", "").strip() or "Unknown ROME",
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="rome_code")
    return df


def _bootstrap_schema(conn) -> None:
    """Execute the gold star schema DDL (idempotent — CREATE IF NOT EXISTS)."""
    bootstrap_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "postgres", "init", "003_gold_star_schema.sql"
    )
    with open(bootstrap_path, "r", encoding="utf-8") as f:
        bootstrap_sql = f.read()
    sql_no_comments = re.sub(r"--[^\n]*", "", bootstrap_sql)
    with conn.cursor() as cur:
        for stmt in sql_no_comments.split(";"):
            if stmt.strip():
                cur.execute(stmt)
    conn.commit()


def load_rome_dim() -> dict:
    """Upsert gold.dim_code_rome from the France Travail ROME JSONL.

    Returns:
        Dict with upserted row count.
    """
    if psycopg is None:
        raise RuntimeError("psycopg is required")

    load_project_env()
    dsn = os.getenv("JOBSTORE_DSN")
    if not dsn:
        raise ValueError("JOBSTORE_DSN is not set")

    df = _build_rome_frame()
    logger.info("ROME frame built: %d codes", len(df))

    upserted = 0
    with psycopg.connect(dsn, autocommit=False) as conn:
        _bootstrap_schema(conn)
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO gold.dim_code_rome (rome_code, rome_label)
                    VALUES (%s, %s)
                    ON CONFLICT (rome_code) DO UPDATE SET
                        rome_label = EXCLUDED.rome_label,
                        updated_at = NOW()
                    """,
                    (row["rome_code"], row["rome_label"]),
                )
                upserted += 1
        conn.commit()

    logger.info("dim_code_rome upserted: %d rows", upserted)
    return {"upserted": upserted}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = load_rome_dim()
    logger.info("ROME dim load completed: %s", result)


if __name__ == "__main__":
    main()
