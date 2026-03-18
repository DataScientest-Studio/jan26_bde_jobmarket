"""Star schema loader: staging + 3 dimensions + minimal fact table.

Load pipeline:
1. Read Silver datasets according to the selected source mode (see below).
2. Normalize records into a run-scoped staging table in PostgreSQL (gold.stg_offer).
3. Upsert 3 dimensions: ROME (dim_code_rome), contract (dim_type_contrat), experience (dim_experience).
4. Upsert the fact table (fact_offre_emploi): 1 row per offer + source.

---------------------------------------------------------------------
Source modes (--source-mode CLI argument / source_mode parameter)
---------------------------------------------------------------------

  auto (default)
      Prefer silver/status_analytics (segment=complete_dataset).
      If no file is found there, automatically fall back to
      silver/merged + silver/status_history.
      Recommended for most runs.

  complete
      Use only the silver/status_analytics dataset
      (segment=complete_dataset). This dataset is produced by
      generate_status_evolution_datasets.py and carries enriched
      temporal status per offer (published_at, unpublished_at,
      reappeared_at, etc.).
      Use this when the status_analytics pipeline has already run.

  fallback
      Join the latest silver/merged file with the latest
      silver/status_history snapshot. Less rich in temporal metadata,
      but always available even when the status_analytics pipeline
      has not run yet.

---------------------------------------------------------------------
Incremental mode (incremental parameter, always active)
---------------------------------------------------------------------

  True (current behaviour)
      Each dataset is identified by a deterministic run_id
      (gold-load-<YYYYMMDD>-<source>). Before importing, the loader
      checks gold.imported_snapshots: if the run_id is already present
      the dataset is skipped. Only new snapshots are loaded, making
      every execution idempotent.

  False (not implemented — reserved)
      Would force a full reload of all datasets regardless of prior
      imports. Not available yet: manually DELETE the relevant row(s)
      from gold.imported_snapshots to replay a specific snapshot.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.config.env import load_project_env
from src.storage.storage import get_storage_from_env
from src.utils import storage_tools

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None


logger = logging.getLogger(__name__)


def _extract_dt_from_key(key: str, pattern: str = r"dt=(\d{4}-\d{2}-\d{2})") -> Optional[str]:
    match = re.search(pattern, key)
    return match.group(1) if match else None


def _extract_merged_dt_from_key(key: str) -> Optional[str]:
    merged = re.search(r"merged_dt=(\d{4}-\d{2}-\d{2})", key)
    if merged:
        return merged.group(1)
    return _extract_dt_from_key(key)


def _latest_complete_dataset() -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    storage_analytics = get_storage_from_env("silver", "status_analytics")
    keys = [k for k in storage_analytics.list_keys("") if k.endswith(".parquet") and "/segment=complete_dataset/" in k]
    if not keys:
        return None, None, None

    dated_keys = []
    for key in keys:
        dt = _extract_dt_from_key(key)
        if dt:
            dated_keys.append((dt, key))

    if not dated_keys:
        return None, None, None

    dated_keys.sort(key=lambda x: x[0])
    snapshot_dt, latest_key = dated_keys[-1]
    df = storage_tools.load_parquet_dataset(storage_analytics, latest_key)
    if df is None or df.empty:
        return None, None, None

    return df, snapshot_dt, latest_key


def _latest_status_and_merged() -> Tuple[pd.DataFrame, str, str]:
    storage_merged = get_storage_from_env("silver", "merged")
    storage_status = get_storage_from_env("silver", "status_history")

    merged_keys = [k for k in storage_merged.list_keys("") if k.endswith(".parquet")]
    if not merged_keys:
        raise RuntimeError("No parquet file found in silver/merged")

    merged_dated = []
    for key in merged_keys:
        dt = _extract_merged_dt_from_key(key)
        if dt:
            merged_dated.append((dt, key))

    if not merged_dated:
        raise RuntimeError("No merged_dt/dt found in silver/merged keys")

    merged_dated.sort(key=lambda x: x[0])
    merged_dt, merged_key = merged_dated[-1]

    status_keys = [k for k in storage_status.list_keys("") if k.endswith(".parquet") and "segment=offer_status" in k]
    if not status_keys:
        raise RuntimeError("No status snapshot found in silver/status_history")

    status_dated = []
    for key in status_keys:
        dt = _extract_dt_from_key(key)
        if dt:
            status_dated.append((dt, key))

    if not status_dated:
        raise RuntimeError("No dt=... found in silver/status_history keys")

    status_dated.sort(key=lambda x: x[0])
    status_dt, status_key = status_dated[-1]

    df_merged = storage_tools.load_parquet_dataset(storage_merged, merged_key)
    df_status = storage_tools.load_parquet_dataset(storage_status, status_key)

    if df_merged is None or df_merged.empty:
        raise RuntimeError(f"Unable to read latest merged dataset: {merged_key}")
    if df_status is None or df_status.empty:
        raise RuntimeError(f"Unable to read latest status dataset: {status_key}")

    status_cols = [
        "id",
        "source",
        "status",
        "published_at",
        "unpublished_at",
        "reappeared_at",
        "first_seen_dt",
        "last_seen_dt",
    ]
    available_status_cols = [c for c in status_cols if c in df_status.columns]

    df = df_merged.merge(
        df_status[available_status_cols],
        on=["id", "source"],
        how="left",
        suffixes=("", "_status"),
    )

    # Prefer status-side values when available.
    if "published_at_status" in df.columns:
        df["published_at"] = df["published_at_status"].combine_first(df.get("published_at"))
        df = df.drop(columns=["published_at_status"])

    snapshot_dt = max(merged_dt, status_dt)
    return df, snapshot_dt, f"merged={merged_key} + status={status_key}"


def _prepare_staging_frame(df: pd.DataFrame, snapshot_dt: str, run_id: str) -> pd.DataFrame:
    working = df.copy()

    rename_map = {
        "id": "offer_id",
    }
    working = working.rename(columns=rename_map)

    required_cols = {
        "offer_id": "",
        "source": "UNKNOWN",
        "title": "",
        "description": "",
        "status": "UNKNOWN",
        "published_at": None,
        "updated_at": None,
        "unpublished_at": None,
        "rome_code": "UNKNOWN",
        "rome_label": "Unknown ROME",
        "contract_normalized": "UNKNOWN",
        "contract_detail": None,
        "contract_type": "UNKNOWN",
        "worktime": "UNKNOWN",
        "experience_normalized": "UNKNOWN",
        "experience_index": None,
        "experience_detail": None,
        "experience_level": "UNKNOWN",
        "experience_description": "UNKNOWN",
        "job_postal_code": "",
        "job_city": "",
        "company_name": "",
        "company_city": "",
        "company_postal_code": "",
        "naf_code": "UNKNOWN",
        "salary_min": 0.0,
        "salary_max": 0.0,
        "salary_periodicity": "UNKNOWN",
        "periodicity": None,
        "yearly_min": None,
        "yearly_max": None,
        "salary_min_computed": None,
        "salary_max_computed": None,
    }

    for col, default_value in required_cols.items():
        if col not in working.columns:
            working[col] = default_value

    staging_cols = list(required_cols.keys())
    staging = working[staging_cols].copy()

    # Prioritize normalized source columns; fallback to legacy fields when missing.
    staging["contract_normalized"] = (
        staging["contract_normalized"].fillna(staging["contract_type"]).replace("", pd.NA)
    )
    staging["experience_normalized"] = (
        staging["experience_normalized"]
        .fillna(staging["experience_level"])
        .fillna(staging["experience_description"])
        .replace("", pd.NA)
    )

    # Standardize nullish text fields to deterministic values for dimension joins.
    text_cols = [
        "rome_code",
        "rome_label",
        "contract_normalized",
        "contract_type",
        "worktime",
        "experience_normalized",
        "experience_level",
        "experience_description",
        "source",
        "status",
        "naf_code",
    ]
    for col in text_cols:
        staging[col] = staging[col].fillna("UNKNOWN").astype(str).str.strip()
        staging[col] = staging[col].replace("", "UNKNOWN")

    # Keep a single business label for missing experience values.
    staging["experience_normalized"] = staging["experience_normalized"].replace(
        {"UNKNOWN": "Non précisé", "NAN": "Non précisé"}
    )
    # Keep a single business label for missing/non-contract values.
    staging["contract_normalized"] = staging["contract_normalized"].replace(
        {
            "UNKNOWN": "Non précisé",
            "Inconnu": "Non précisé",
            "NAN": "Non précisé",
            "other": "Non précisé",
            "full_time": "Non précisé",
            "part_time": "Non précisé",
        }
    )

    for dt_col in ["published_at", "updated_at", "unpublished_at"]:
        staging[dt_col] = pd.to_datetime(staging[dt_col], errors="coerce", utc=True)

    for num_col in ["salary_min", "salary_max"]:
        staging[num_col] = pd.to_numeric(staging[num_col], errors="coerce").fillna(0.0)

    # COPY target column is INT; convert float-like values (e.g. 0.0) to nullable Int.
    staging["experience_index"] = pd.to_numeric(staging["experience_index"], errors="coerce").astype("Int64")

    staging["snapshot_dt"] = pd.to_datetime(snapshot_dt).date()
    staging["run_id"] = run_id

    ordered_cols = [
        "run_id",
        "snapshot_dt",
        "offer_id",
        "source",
        "title",
        "description",
        "status",
        "published_at",
        "updated_at",
        "unpublished_at",
        "rome_code",
        "rome_label",
        "contract_normalized",
        "contract_detail",
        "contract_type",
        "worktime",
        "experience_normalized",
        "experience_index",
        "experience_detail",
        "experience_level",
        "experience_description",
        "job_postal_code",
        "job_city",
        "company_name",
        "company_city",
        "company_postal_code",
        "naf_code",
        "salary_min",
        "salary_max",
        "salary_periodicity",
        "periodicity",
        "yearly_min",
        "yearly_max",
        "salary_min_computed",
        "salary_max_computed",
    ]

    # Deduplicate by offer key for the run.
    staging = staging[ordered_cols].drop_duplicates(subset=["offer_id", "source"], keep="last")
    return staging


def _copy_staging(conn, df: pd.DataFrame, chunk_size: int = 50_000) -> None:
    cols = [
        "run_id",
        "snapshot_dt",
        "offer_id",
        "source",
        "title",
        "description",
        "status",
        "published_at",
        "updated_at",
        "unpublished_at",
        "rome_code",
        "rome_label",
        "contract_normalized",
        "contract_detail",
        "contract_type",
        "worktime",
        "experience_normalized",
        "experience_index",
        "experience_detail",
        "experience_level",
        "experience_description",
        "job_postal_code",
        "job_city",
        "company_name",
        "company_city",
        "company_postal_code",
        "naf_code",
        "salary_min",
        "salary_max",
        "salary_periodicity",
        "periodicity",
        "yearly_min",
        "yearly_max",
        "salary_min_computed",
        "salary_max_computed",
    ]

    copy_sql = (
        "COPY gold.stg_offer ("
        + ", ".join(cols)
        + ") FROM STDIN WITH (FORMAT CSV, NULL '')"
    )

    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            with tqdm(total=len(df), unit="rows", desc="COPY staging") as pbar:
                for start in range(0, len(df), chunk_size):
                    chunk = df.iloc[start : start + chunk_size]
                    buf = io.StringIO()
                    chunk.to_csv(buf, index=False, header=False, columns=cols, na_rep="")
                    copy.write(buf.getvalue())
                    pbar.update(len(chunk))


def _upsert_dimensions_and_fact(conn, run_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO gold.dim_geo (code_postal, code_departement, nom_departement, code_region, nom_region)
            SELECT DISTINCT
                COALESCE(NULLIF(job_postal_code, ''), 'UNKNOWN') AS code_postal,
                'UNKNOWN' AS code_departement,
                'Unknown' AS nom_departement,
                'UNKNOWN' AS code_region,
                'Unknown' AS nom_region
            FROM gold.stg_offer
            WHERE run_id = %s
            ON CONFLICT (code_postal)
            DO UPDATE SET updated_at = NOW()
            """,
            (run_id,),
        )

        cur.execute(
            """
            INSERT INTO gold.dim_naf (naf_code, naf_label)
            SELECT DISTINCT
                COALESCE(NULLIF(naf_code, ''), 'UNKNOWN') AS naf_code,
                'Unknown NAF' AS naf_label
            FROM gold.stg_offer
            WHERE run_id = %s
            ON CONFLICT (naf_code)
            DO UPDATE SET updated_at = NOW()
            """,
            (run_id,),
        )

        cur.execute(
            """
            INSERT INTO gold.dim_code_rome (rome_code, rome_label)
            SELECT
                rome_code,
                MAX(rome_label) AS rome_label
            FROM (
                SELECT
                    COALESCE(NULLIF(rome_code, ''), 'UNKNOWN') AS rome_code,
                    COALESCE(NULLIF(rome_label, ''), 'Unknown ROME') AS rome_label
                FROM gold.stg_offer
                WHERE run_id = %s
            ) dedup
            GROUP BY rome_code
            ON CONFLICT (rome_code)
            DO UPDATE SET
                rome_label = EXCLUDED.rome_label,
                updated_at = NOW()
            """,
            (run_id,),
        )

        cur.execute(
            """
            INSERT INTO gold.dim_type_contrat (contract_type)
            SELECT DISTINCT
                COALESCE(NULLIF(contract_normalized, ''), 'Non précisé') AS contract_type
            FROM gold.stg_offer
            WHERE run_id = %s
            ON CONFLICT (contract_type)
            DO UPDATE SET
                updated_at = NOW()
            """,
            (run_id,),
        )

        cur.execute(
            """
            INSERT INTO gold.dim_experience (experience_level)
            SELECT DISTINCT
                COALESCE(NULLIF(experience_normalized, ''), 'Non précisé') AS experience_level
            FROM gold.stg_offer
            WHERE run_id = %s
            ON CONFLICT (experience_level)
            DO UPDATE SET
                updated_at = NOW()
            """,
            (run_id,),
        )

        cur.execute(
                """
                INSERT INTO gold.fact_offre_emploi (
                        offer_id,
                        source,
                        snapshot_dt,
                        geo_key,
                        naf_key,
                        rome_key,
                        contract_key,
                        experience_key,
                        status,
                        published_at,
                        updated_at,
                        unpublished_at,
                        title,
                        description,
                        job_postal_code,
                        job_city,
                        company_name,
                        company_city,
                        company_postal_code,
                        salary_min,
                        salary_max,
                        salary_periodicity,
                        salary_min_computed,
                        salary_max_computed,
                        load_run_id
                )
                SELECT
                        s.offer_id,
                        s.source,
                        s.snapshot_dt,
                        dg.geo_key,
                        dn.naf_key,
                        dr.rome_key,
                        dc.contract_key,
                        de.experience_key,
                        s.status,
                        s.published_at,
                        s.updated_at,
                        s.unpublished_at,
                        s.title,
                        s.description,
                        s.job_postal_code,
                        s.job_city,
                        s.company_name,
                        s.company_city,
                        s.company_postal_code,
                        s.salary_min,
                        s.salary_max,
                        s.salary_periodicity,
                        s.salary_min_computed,
                        s.salary_max_computed,
                        s.run_id
                FROM gold.stg_offer s
                JOIN gold.dim_geo dg
                    ON dg.code_postal = COALESCE(NULLIF(s.job_postal_code, ''), 'UNKNOWN')
                JOIN gold.dim_naf dn
                    ON dn.naf_code = COALESCE(NULLIF(s.naf_code, ''), 'UNKNOWN')
                JOIN gold.dim_code_rome dr
                    ON dr.rome_code = COALESCE(NULLIF(s.rome_code, ''), 'UNKNOWN')
                JOIN gold.dim_type_contrat dc
                    ON dc.contract_type = COALESCE(NULLIF(s.contract_normalized, ''), 'Non précisé')
                JOIN gold.dim_experience de
                    ON de.experience_level = COALESCE(NULLIF(s.experience_normalized, ''), 'Non précisé')
                WHERE s.run_id = %s
                ON CONFLICT (offer_id, source)
                DO UPDATE SET
                        snapshot_dt = EXCLUDED.snapshot_dt,
                        geo_key = EXCLUDED.geo_key,
                        naf_key = EXCLUDED.naf_key,
                        rome_key = EXCLUDED.rome_key,
                        contract_key = EXCLUDED.contract_key,
                        experience_key = EXCLUDED.experience_key,
                        status = EXCLUDED.status,
                        published_at = EXCLUDED.published_at,
                        updated_at = EXCLUDED.updated_at,
                        unpublished_at = EXCLUDED.unpublished_at,
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        job_postal_code = EXCLUDED.job_postal_code,
                        job_city = EXCLUDED.job_city,
                        company_name = EXCLUDED.company_name,
                        company_city = EXCLUDED.company_city,
                        company_postal_code = EXCLUDED.company_postal_code,
                        salary_min = EXCLUDED.salary_min,
                        salary_max = EXCLUDED.salary_max,
                        salary_periodicity = EXCLUDED.salary_periodicity,
                        salary_min_computed = EXCLUDED.salary_min_computed,
                        salary_max_computed = EXCLUDED.salary_max_computed,
                        load_run_id = EXCLUDED.load_run_id,
                        load_ts = NOW()
                """,
                (run_id,),
        )


def _bootstrap_sql(conn) -> None:
    """Execute the gold star schema DDL (idempotent — CREATE IF NOT EXISTS).

    psycopg3 does not allow multiple statements in a single cursor.execute()
    call. Single-line comments are stripped before splitting on ';' to avoid
    cutting inside comment text that contains semicolons.
    """
    bootstrap_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "postgres", "init", "003_gold_star_schema.sql")
    with open(bootstrap_path, "r", encoding="utf-8") as f:
        sql = f.read()
    sql_no_comments = re.sub(r"--[^\n]*", "", sql)
    with conn.cursor() as cur:
        for stmt in sql_no_comments.split(";"):
            if stmt.strip():
                cur.execute(stmt)
    conn.commit()



def run_loader(run_id: Optional[str] = None, source_mode: str = "auto", incremental: bool = True) -> dict:
    """
    Load the Gold star schema from Silver datasets.

    Args:
        run_id: Optional run identifier (auto-generated if None).
        source_mode: Silver data reading strategy.
            - "auto"     : status_analytics preferred, falls back to merged+status_history.
            - "complete" : status_analytics only (segment=complete_dataset).
            - "fallback" : merged + status_history only.
        incremental: If True (default), datasets already recorded in
            gold.imported_snapshots are skipped (idempotent execution).

    Returns:
        Dict with metrics from the last imported snapshot (staging_rows, fact_rows, etc.),
        or {} if no new dataset was imported.
    """
    if psycopg is None:
        raise RuntimeError("psycopg is required to run this loader")

    load_project_env()
    dsn = os.getenv("JOBSTORE_DSN")
    if not dsn:
        raise ValueError("JOBSTORE_DSN is not set")

    # 1. Scan all available datasets (status_analytics preferred)
    datasets = []
    storage_analytics = get_storage_from_env("silver", "status_analytics")
    keys = [k for k in storage_analytics.list_keys("") if k.endswith(".parquet") and "/segment=complete_dataset/" in k]
    for key in keys:
        dt = _extract_dt_from_key(key)
        if dt:
            datasets.append((dt, key, "status_analytics"))

    # Fallback: merged+status_history
    if not datasets:
        storage_merged = get_storage_from_env("silver", "merged")
        merged_keys = [k for k in storage_merged.list_keys("") if k.endswith(".parquet")]
        for key in merged_keys:
            dt = _extract_merged_dt_from_key(key)
            if dt:
                datasets.append((dt, key, "merged"))

    if not datasets:
        raise RuntimeError("No datasets found in silver layers")

    datasets.sort(key=lambda x: x[0])

    # 2. Connect, bootstrap schema, and get already imported run_ids
    with psycopg.connect(dsn, autocommit=False) as conn:
        _bootstrap_sql(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT run_id FROM gold.imported_snapshots")
            already_imported = set(r[0] for r in cur.fetchall())

    results = []
    for dt, key, source in datasets:
        # Compose a run_id for this import
        run_id = f"gold-load-{dt.replace('-', '')}-{source}"
        if run_id in already_imported:
            logger.info(f"[SKIP] Dataset {key} (run_id={run_id}) already imported.")
            continue

        logger.info(f"[IMPORT] Dataset {key} (run_id={run_id}) ...")
        # Load dataset
        if source == "status_analytics":
            dataset = storage_tools.load_parquet_dataset(storage_analytics, key)
        else:
            storage_merged = get_storage_from_env("silver", "merged")
            dataset = storage_tools.load_parquet_dataset(storage_merged, key)
        if dataset is None or dataset.empty:
            logger.warning(f"[SKIP] Dataset {key} is empty or unreadable.")
            continue

        snapshot_dt = dt
        source_label = key
        staging_df = _prepare_staging_frame(dataset, snapshot_dt=snapshot_dt, run_id=run_id)
        logger.info(f"Staging rows: {len(staging_df):,}")

        steps = [
            "Bootstrap schema",
            "Clear staging",
            "COPY staging",
            "Upsert dimensions & fact",
            "Count rows",
            "Commit",
            "Purge staging",
        ]
        with tqdm(total=len(steps), unit="step", desc=f"Gold load {run_id}") as pbar:
            with psycopg.connect(dsn, autocommit=False) as conn:
                pbar.set_postfix_str("Bootstrap schema")
                _bootstrap_sql(conn)
                pbar.update(1)

                pbar.set_postfix_str("Clear staging")
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM gold.stg_offer WHERE run_id = %s", (run_id,))
                pbar.update(1)

                pbar.set_postfix_str("COPY staging")
                _copy_staging(conn, staging_df)
                pbar.update(1)

                pbar.set_postfix_str("Upsert dimensions & fact")
                _upsert_dimensions_and_fact(conn, run_id)
                pbar.update(1)

                pbar.set_postfix_str("Count rows")
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM gold.fact_offre_emploi")
                    fact_count = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM gold.dim_code_rome")
                    dim_rome_count = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM gold.dim_type_contrat")
                    dim_contract_count = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM gold.dim_experience")
                    dim_experience_count = cur.fetchone()[0]
                pbar.update(1)

                pbar.set_postfix_str("Commit")
                conn.commit()
                pbar.update(1)

                # Log import in gold.imported_snapshots
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO gold.imported_snapshots (run_id, snapshot_dt, source) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (run_id, snapshot_dt, source)
                    )
                    conn.commit()

                # Purge staging rows for this run.
                #
                # Design choice: stg_offer is a transit zone, not a storage layer.
                # Once a run is committed to the fact/dimension tables and recorded
                # in imported_snapshots, the staging rows are redundant:
                #   - The Silver parquet file is the authoritative source of truth.
                #   - imported_snapshots.source points to the exact parquet key,
                #     allowing a full replay at any time.
                #
                # Trade-off: past staging rows cannot be inspected directly in SQL
                # after the run. To debug a historical load, reload the parquet
                # identified by imported_snapshots.source and re-run _prepare_staging_frame.
                pbar.set_postfix_str("Purge staging")
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM gold.stg_offer WHERE run_id = %s", (run_id,))
                conn.commit()
                pbar.update(1)

        results.append({
            "run_id": run_id,
            "snapshot_dt": snapshot_dt,
            "source": source_label,
            "staging_rows": int(len(staging_df)),
            "fact_rows": int(fact_count),
            "dim_code_rome_rows": int(dim_rome_count),
            "dim_type_contrat_rows": int(dim_contract_count),
            "dim_experience_rows": int(dim_experience_count),
        })

    return results[-1] if results else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Load star schema from silver datasets")
    parser.add_argument("--run-id", default=None, help="Optional run identifier")
    parser.add_argument(
        "--source-mode",
        choices=["auto", "complete", "fallback"],
        default="auto",
        help="Data source mode: complete uses status_analytics complete dataset, fallback uses merged+status_history",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    result = run_loader(run_id=args.run_id, source_mode=args.source_mode)
    logger.info("Star schema load completed: %s", result)


if __name__ == "__main__":
    main()

