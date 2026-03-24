"""
Build the ROME classification training dataset from Silver France Travail normalized data.

Reads the normalized FT Parquet file from the Silver layer (dt={dt}/all.parquet),
filters out ROME classes with fewer than MIN_CLASS_COUNT examples, and writes the
result to the Gold layer under datasets/{dt}/rome_dataset.parquet.

Why Silver FT (not Bronze, not Silver merged)?
- Bronze FT: raw JSONL with unclean text — HTML not stripped, normalization not applied.
- Silver FT: cleaned text (HTML stripped, normalized), rome_code = ground truth from
  France Travail (not a model prediction).

Output schema:
    id        (str)  — original offer identifier
    text      (str)  — concatenated title + description (pre-cleaned by Silver pipeline)
    rome_code (str)  — ROME classification code (label, ground truth from FT)

Usage:
    # Auto mode — picks the latest available dt in Silver FT
    python -m src.data.make_dataset_from_silver

    # Explicit dt
    python -m src.data.make_dataset_from_silver --dt 2026-02-13
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from src.config.env import load_project_env

load_project_env()  # idempotent — safe to call multiple times

from src.storage.storage import get_storage_from_env
from src.utils.storage_tools import get_last_dt_from_storage

logger = logging.getLogger(__name__)

MIN_CLASS_COUNT = int(os.getenv("MIN_CLASS_COUNT", "50"))

# Output key template — dt is injected at runtime
OUTPUT_KEY_TEMPLATE = "datasets/{dt}/rome_dataset.parquet"

# Silver FT key template — written by normalize_ft_jobs as dt={dt}/all.parquet
SILVER_KEY_TEMPLATE = "dt={dt}/all.parquet"


def resolve_dt(storage_silver, dt_arg: str | None) -> str:
    """Return the dt to process: explicit value or latest available in Silver FT."""
    if dt_arg and dt_arg not in ("", "latest"):
        return dt_arg
    latest = get_last_dt_from_storage(storage_silver, "")
    if not latest:
        raise RuntimeError(
            "No Silver FT data found. Run normalize_ft_jobs first."
        )
    logger.info("Auto mode — latest Silver FT dt: %s", latest)
    return latest


def build_dataset(dt: str) -> dict:
    """
    Load Silver FT data for the given dt, filter low-support ROME classes,
    and write the training dataset to Gold.

    Args:
        dt: Partition date string (e.g. '2026-02-13').

    Returns:
        Dict with dt, rows_total, rows_kept, classes_kept, output_key.
    """
    storage_silver = get_storage_from_env("silver", "france_travail")
    storage_gold = get_storage_from_env("gold")

    silver_key = SILVER_KEY_TEMPLATE.format(dt=dt)
    logger.info("Reading Silver FT: %s", silver_key)
    df = storage_silver.read_parquet(silver_key)
    logger.info("Silver FT loaded: %d rows", len(df))

    # Keep only rows with a valid ROME code and non-empty text
    df = df[df["rome_code"].notna() & (df["rome_code"] != "")].copy()

    # Build text field from pre-cleaned title + description
    df["text"] = (
        df["title"].fillna("").apply(lambda t: f"[TITRE] {t}" if t else "") + "\n" +
        df["description"].fillna("").apply(lambda d: f"[DESC] {d}" if d else "")
    ).str.strip()

    df = df[df["text"] != ""].copy()

    rows_total = len(df)
    logger.info("Rows with valid rome_code and text: %d", rows_total)

    if rows_total == 0:
        raise RuntimeError(
            f"No usable rows in Silver FT for dt={dt}. "
            "Check that normalize_ft_jobs completed successfully."
        )

    # Drop ROME classes with fewer than MIN_CLASS_COUNT examples
    counts = df["rome_code"].value_counts()
    keep_codes = counts[counts >= MIN_CLASS_COUNT].index
    df = df[df["rome_code"].isin(keep_codes)].reset_index(drop=True)

    logger.info(
        "Classes kept (>= %d examples): %d | Rows kept: %d",
        MIN_CLASS_COUNT, len(keep_codes), len(df),
    )

    # Select output columns only
    out_df = df[["id", "text", "rome_code"]].copy()

    output_key = OUTPUT_KEY_TEMPLATE.format(dt=dt)
    logger.info("Writing dataset to Gold: %s", output_key)
    storage_gold.write_parquet(output_key, out_df)
    logger.info("Done — %d rows written", len(out_df))

    return {
        "dt": dt,
        "rows_total": rows_total,
        "rows_kept": len(out_df),
        "classes_kept": len(keep_codes),
        "output_key": output_key,
    }


def reconstruct_all() -> list[dict]:
    """Rebuild Gold datasets for every dt partition available in Silver FT.

    Iterates all dt= partitions in Silver FT in ascending date order and calls
    build_dataset(dt) for each one. Useful for backfilling the Gold dataset history
    after a schema change or initial setup.

    Returns:
        List of result dicts (one per dt), same structure as build_dataset().
    """
    storage_silver = get_storage_from_env("silver", "france_travail")
    prefixes = storage_silver.list_prefixes("")
    dts = sorted([
        p.split("dt=")[-1].strip("/")
        for p in prefixes
        if "dt=" in p
    ])
    if not dts:
        raise RuntimeError("No Silver FT partitions found. Run normalize_ft_jobs first.")
    logger.info("daily_reconstruction — %d partition(s) to process: %s", len(dts), dts)
    results = []
    for dt in dts:
        logger.info("Processing dt=%s ...", dt)
        result = build_dataset(dt)
        results.append(result)
        logger.info("dt=%s done: %s", dt, result)
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build ROME training dataset from Silver FT normalized data."
    )
    parser.add_argument(
        "--dt",
        default=None,
        help="Partition date to process (YYYY-MM-DD). Omit for auto (latest Silver FT dt). Ignored in daily_reconstruction mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "daily_reconstruction"],
        default="single",
        help="single: process one dt (default). daily_reconstruction: rebuild all available dt partitions.",
    )
    args = parser.parse_args()

    if args.mode == "daily_reconstruction":
        results = reconstruct_all()
        logger.info("Reconstruction completed: %d dataset(s) built.", len(results))
    else:
        storage_silver = get_storage_from_env("silver", "france_travail")
        dt = resolve_dt(storage_silver, args.dt)
        result = build_dataset(dt)
        logger.info("Dataset build completed: %s", result)


if __name__ == "__main__":
    main()
