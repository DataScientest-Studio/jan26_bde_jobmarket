"""Geographic dimension loader: builds gold.dim_geo from INSEE communes CSV.

Source — storage layer bronze/insee (S3/MinIO: bronze/insee/, local: data/bronze/insee/):
  - 20230823-communes-departement-region.csv
    Columns used: code_postal, code_departement, nom_departement,
                  code_region, nom_region, latitude, longitude.

File is read via get_storage_from_env("bronze", "insee") using read_bytes() + BytesIO,
so the same code works against both local filesystem and MinIO/S3.

Grain: one row per code_postal. When multiple communes share the same postal
code, the first occurrence is kept for the label and a centroid is computed
from the available latitude/longitude values.

The UNKNOWN sentinel row is inserted by the schema DDL (ON CONFLICT DO NOTHING).

---------------------------------------------------------------------
Design: flat dimension (star schema)
---------------------------------------------------------------------

dim_geo is intentionally flat: each row contains the postal code together
with its département and région, fully denormalized. The same nom_region
value is repeated across all postal codes in that region.

This redundancy is the core trade-off of star schema (Kimball):

  Pros
  - Any GROUP BY (postal code, département, région) needs exactly one JOIN
    from fact_offre_emploi to dim_geo — no additional JOINs required.
  - Grafana queries stay simple regardless of the aggregation level.

  Cons
  - Minor storage overhead (region/department labels duplicated across rows).
    Acceptable here: dim_geo has ~6 000 rows, negligible vs the fact table.

An alternative snowflake design (fact → dim_postal → dim_departement →
dim_region) would eliminate redundancy but require a multi-level JOIN for
every region-level dashboard panel — more complexity for no analytical gain
at this scale.

Indexing strategy: no indexes on dim_geo columns (13 regions, ~100
departments — PostgreSQL will cache the whole table in memory). Indexes
belong on fact_offre_emploi.geo_key (FK join) and analytical columns such
as snapshot_dt.

Usage:
    python -m src.ingest.gold.load_geo_dim
"""

from __future__ import annotations

import io
import logging
import os

import pandas as pd

from src.config.env import load_project_env
from src.storage.storage import get_storage_from_env

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None

logger = logging.getLogger(__name__)

# S3/MinIO: bronze/insee/
# Local:    data/bronze/insee/
_INSEE_STORAGE_LAYER = "bronze"
_INSEE_STORAGE_SOURCE = "insee"
_CSV_KEY = "20230823-communes-departement-region.csv"


def _build_geo_frame() -> pd.DataFrame:
    """Build a deduplicated dim_geo DataFrame at code_postal grain.

    The INSEE CSV contains multiple communes per postal code (e.g. rural areas
    where several villages share the same code_postal). 
    Since dim_geo is keyed  code_postal, the raw data must be collapsed to one row per postal code
    before loading — otherwise fact_offre_emploi JOINs would produce duplicate and inflate aggregation results.

    Aggregation strategy (groupby code_postal):
    - Label columns (nom_departement, nom_region, etc.): "first" — the first
      commune encountered is used as the representative label for the postal
      code. Arbitrary but consistent; label accuracy is sufficient for
      dashboard-level grouping.
    - Coordinates (latitude, longitude): "mean" — geographic centroid of all
      communes sharing the postal code. 
    """
    storage = get_storage_from_env(_INSEE_STORAGE_LAYER, _INSEE_STORAGE_SOURCE)
    raw = pd.read_csv(io.BytesIO(storage.read_bytes(_CSV_KEY)), dtype=str, sep=None, engine="python")

    cols = {
        "code_postal": "code_postal",
        "code_departement": "code_departement",
        "nom_departement": "nom_departement",
        "nom_commune": "nom_commune",
        "code_region": "code_region",
        "nom_region": "nom_region",
        "latitude": "latitude",
        "longitude": "longitude",
    }
    df = raw[list(cols.keys())].copy()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Aggregate: first label, centroid lat/lon per postal code
    agg = df.groupby("code_postal", sort=False).agg(
        code_departement=("code_departement", "first"),
        nom_departement=("nom_departement", "first"),
        nom_commune=("nom_commune", "first"),
        code_region=("code_region", "first"),
        nom_region=("nom_region", "first"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    ).reset_index()

    for col in ["code_departement", "nom_departement", "code_region", "nom_region"]:
        agg[col] = agg[col].fillna("").str.strip()

    return agg


def _bootstrap_schema(conn) -> None:
    """Execute the gold star schema DDL (idempotent — CREATE IF NOT EXISTS).

    psycopg3 does not allow multiple statements in a single cursor.execute()
    call. Single-line comments are stripped before splitting on ';' to avoid
    cutting inside comment text that contains semicolons.
    """
    import re
    bootstrap_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "postgres", "init", "003_gold_star_schema.sql")
    with open(bootstrap_path, "r", encoding="utf-8") as f:
        bootstrap_sql = f.read()
    sql_no_comments = re.sub(r"--[^\n]*", "", bootstrap_sql)
    with conn.cursor() as cur:
        for stmt in sql_no_comments.split(";"):
            if stmt.strip():
                cur.execute(stmt)
    conn.commit()


def load_geo_dim() -> dict:
    """Upsert gold.dim_geo from the local INSEE communes CSV.

    Returns:
        Dict with upserted row count.
    """
    if psycopg is None:
        raise RuntimeError("psycopg is required")

    load_project_env()
    dsn = os.getenv("JOBSTORE_DSN")
    if not dsn:
        raise ValueError("JOBSTORE_DSN is not set")
    
    print(f"dsn : {dsn}")

    df = _build_geo_frame()
    logger.info("Geo frame built: %d postal codes", len(df))

    upserted = 0
    with psycopg.connect(dsn, autocommit=False) as conn:
        _bootstrap_schema(conn)
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                #print(f"Upserting {row['code_postal']} - {row['nom_commune']} ({row['nom_departement']}, {row['nom_region']})")
                cur.execute(
                    """
                    INSERT INTO gold.dim_geo (
                        code_postal, code_departement, nom_departement,
                        code_region, nom_region, nom_commune, latitude, longitude
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (code_postal) DO UPDATE SET
                        code_departement = EXCLUDED.code_departement,
                        nom_departement  = EXCLUDED.nom_departement,
                        code_region      = EXCLUDED.code_region,
                        nom_region       = EXCLUDED.nom_region,
                        nom_commune      = EXCLUDED.nom_commune,
                        latitude         = EXCLUDED.latitude,
                        longitude        = EXCLUDED.longitude,
                        updated_at       = NOW()
                    """,
                    (
                        row["code_postal"],
                        row["code_departement"] or None,
                        row["nom_departement"] or None,
                        row["code_region"] or None,
                        row["nom_region"] or None,
                        row["nom_commune"] or None,
                        None if pd.isna(row["latitude"]) else float(row["latitude"]),
                        None if pd.isna(row["longitude"]) else float(row["longitude"]),
                    ),
                )
                upserted += 1
        conn.commit()

    logger.info("dim_geo upserted: %d rows", upserted)
    return {"upserted": upserted}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = load_geo_dim()
    logger.info("Geo dim load completed: %s", result)


if __name__ == "__main__":
    main()
