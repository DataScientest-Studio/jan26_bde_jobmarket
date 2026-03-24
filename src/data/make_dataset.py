"""
Build the ROME classification training dataset from Bronze France Travail offers.

Reads raw JSONL offer files from the Bronze layer (MinIO or local), extracts
title + description + competences into a single text field, filters out ROME
classes with fewer than MIN_CLASS_COUNT examples, and writes the result as a
Parquet file to the Gold layer under datasets/{dt}/rome_dataset.parquet.

Output schema:
    id        (str)  — original offer identifier
    text      (str)  — concatenated title / description / competences
    rome_code (str)  — ROME classification code (label)

Usage:
    # Auto mode — picks the latest available dt in Bronze FT
    python -m src.data.make_dataset

    # Explicit dt
    python -m src.data.make_dataset --dt 2026-02-13
"""

import argparse
import logging
import os
import re
from typing import Any, Dict, List, Iterable, Optional

import pandas as pd

from src.config.env import load_project_env
load_project_env()  # idempotent — safe to call multiple times

from src.storage import get_storage_from_env
from src.utils.storage_tools import get_last_dt_from_storage

logger = logging.getLogger(__name__)

# Output key template — dt is injected at runtime
OUTPUT_KEY_TEMPLATE = "datasets/{dt}/rome_dataset.parquet"

# Bronze FT offers: offers/dt={dt}/run_id={run_id}/code_rome=.../.../*.jsonl
BRONZE_PREFIX_TEMPLATE = "offers/dt={dt}/run_id={run_id}"
BRONZE_OFFERS_ROOT = "offers/"

MIN_CLASS_COUNT = int(os.getenv("MIN_CLASS_COUNT", "50"))
MAX_COMPETENCES = int(os.getenv("MAX_COMPETENCES", "25"))

WHITESPACE_RE = re.compile(r"\s+")

def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = WHITESPACE_RE.sub(" ", s)
    return s.strip()

def extract_competences(record: Dict[str, Any], max_items: int = 25) -> List[str]:
    raw = record.get("competences", [])
    out: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                lib = clean_text(item.get("libelle"))
                if lib:
                    out.append(lib)

    # Deduplicate while preserving original order
    seen = set()
    dedup = []
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(x)
    return dedup[:max_items]

def build_text_field(record: Dict[str, Any]) -> str:
    title = clean_text(record.get("intitule"))
    desc = clean_text(record.get("description"))
    comps = extract_competences(record, MAX_COMPETENCES)

    parts = []
    if title:
        parts.append(f"[TITRE] {title}")
    if desc:
        parts.append(f"[DESC] {desc}")
    if comps:
        parts.append("[COMP] " + " ".join(comps))

    return "\n".join(parts).strip()

def resolve_dt(storage_bronze, dt_arg: str | None) -> str:
    """Return the dt to process: explicit value or latest available in Bronze FT."""
    if dt_arg and dt_arg not in ("", "latest"):
        return dt_arg
    latest = get_last_dt_from_storage(storage_bronze, BRONZE_OFFERS_ROOT)
    if not latest:
        raise RuntimeError(
            "No Bronze FT data found. Run the FT ingestion pipeline first."
        )
    logger.info("Auto mode — latest Bronze FT dt: %s", latest)
    return latest


def resolve_run_id(storage_bronze, dt: str) -> str:
    """Return the latest run_id under offers/dt={dt}/.

    Multiple run_ids may exist under the same dt partition (e.g. a partial run
    followed by a complete re-run). Only the most recent run_id is used to
    avoid reading duplicate or stale data.
    """
    prefix = f"offers/dt={dt}/"
    prefixes = storage_bronze.list_prefixes(prefix)
    run_ids = [
        p.split("run_id=")[-1].strip("/")
        for p in prefixes
        if "run_id=" in p
    ]
    if not run_ids:
        raise RuntimeError(
            f"No run_id found under {prefix}. "
            "Check that the FT ingestion pipeline completed successfully."
        )
    latest = max(run_ids)
    logger.info("Latest run_id for dt=%s: %s", dt, latest)
    return latest


def iter_bronze_offers(storage, bronze_prefix: str) -> Iterable[Dict[str, Any]]:
    """Yield raw offer records from all JSONL files under bronze_prefix."""
    keys = storage.list_keys(bronze_prefix)
    keys = [k for k in keys if k.endswith(".jsonl")]
    for key in keys:
        for rec in storage.read_jsonl(key):
            yield rec


def build_dataset(dt: str) -> dict:
    """
    Load Bronze FT offers for the given dt, build text fields, filter low-support
    ROME classes, and write the training dataset to Gold.

    Args:
        dt: Partition date string (e.g. '2026-02-13').

    Returns:
        Dict with dt, rows_total, rows_kept, classes_kept, output_key.
    """
    storage_bronze = get_storage_from_env("bronze", "france_travail")
    storage_gold = get_storage_from_env("gold")

    run_id = resolve_run_id(storage_bronze, dt)
    bronze_prefix = BRONZE_PREFIX_TEMPLATE.format(dt=dt, run_id=run_id)
    logger.info("Reading Bronze FT offers from prefix: %s", bronze_prefix)

    rows = []
    total = 0
    kept = 0
    for rec in iter_bronze_offers(storage_bronze, bronze_prefix):
        total += 1

        rome = rec.get("romeCode")
        if not rome:
            continue

        text = build_text_field(rec)
        if not text:
            continue

        rows.append({"id": rec.get("id"), "text": text, "rome_code": rome})
        kept += 1

        if total % 5000 == 0:
            print(f"\r📦 Records seen: {total} | Rows kept: {kept}", end="")

    df = pd.DataFrame(rows)
    print(f"\n📦 Records seen (job offers): {total}")
    print(f"📊 Rows kept (pre-filter): {len(df)}")

    if len(df) == 0:
        raise RuntimeError(
            f"No training rows produced for dt={dt}. "
            "Check that the FT ingestion pipeline completed successfully."
        )

    counts = df["rome_code"].value_counts()

    # Drop ROME classes with fewer than MIN_CLASS_COUNT examples — ensures minimum
    # viable support per class for training and evaluation splits
    keep_codes = counts[counts >= MIN_CLASS_COUNT].index
    df = df[df["rome_code"].isin(keep_codes)].reset_index(drop=True)

    logger.info(
        "Classes kept (>= %d examples): %d | Rows kept: %d",
        MIN_CLASS_COUNT, len(keep_codes), len(df),
    )
    print(f"📌 Classes kept (# elements >= {MIN_CLASS_COUNT}): {len(keep_codes)} / 1584 ROME codes")
    print(f"📊 Rows kept (post-filter): {len(df)}")

    output_key = OUTPUT_KEY_TEMPLATE.format(dt=dt)
    print(f"💾 Writing dataset to: {output_key}")
    storage_gold.write_parquet(output_key, df)
    print("✅ make_dataset — done")

    return {
        "dt": dt,
        "run_id": run_id,
        "rows_total": total,
        "rows_kept": len(df),
        "classes_kept": len(keep_codes),
        "output_key": output_key,
    }


def reconstruct_all() -> list[dict]:
    """Rebuild Gold datasets for every dt partition available in Bronze FT.

    Iterates all dt= partitions under offers/ in ascending date order and calls
    build_dataset(dt) for each one. Useful for backfilling the Gold dataset history
    after a schema change or initial setup.

    Returns:
        List of result dicts (one per dt), same structure as build_dataset().
    """
    storage_bronze = get_storage_from_env("bronze", "france_travail")
    prefixes = storage_bronze.list_prefixes(BRONZE_OFFERS_ROOT)
    dts = sorted([
        p.split("dt=")[-1].strip("/")
        for p in prefixes
        if "dt=" in p
    ])
    if not dts:
        raise RuntimeError("No Bronze FT partitions found. Run the FT ingestion pipeline first.")
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
        description="Build ROME training dataset from Bronze FT raw offers."
    )
    parser.add_argument(
        "--dt",
        default=None,
        help="Partition date to process (YYYY-MM-DD). Omit for auto (latest Bronze FT dt). Ignored in daily_reconstruction mode.",
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
        storage_bronze = get_storage_from_env("bronze", "france_travail")
        dt = resolve_dt(storage_bronze, args.dt)
        result = build_dataset(dt)
        logger.info("Dataset build completed: %s", result)


if __name__ == "__main__":
    main()
