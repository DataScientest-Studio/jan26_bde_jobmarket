"""
===================================================================================
OFFER STATUS TRACKING - SEPARATE IMMUTABLE STATUS DATASET ARCHITECTURE
===================================================================================

ARCHITECTURE & DESIGN PRINCIPLES
=================================

This module implements a **separate status history dataset** pattern for 
tracking job offer lifecycle while maintaining data immutability and reproducibility.

PROBLEM STATEMENT:
------------------
The original approach mutated the merged dataset by injecting unpublished offers,
which:
  ❌ Breaks reproducibility (can't replay history with original state)
  ❌ Mixes source data (offers) with derived state (status changes)
  ❌ Makes incremental updates inefficient (recalculates all offers daily)
  ❌ Prevents isolation of status logic from offer merging logic

SOLUTION: SEPARATE STATUS DATASET
----------------------------------
Instead of modifying the merged dataset, we maintain a separate, immutable 
status_history dataset that tracks offer lifecycle events:

  Directory: silver/status_history/
  Partitioning: dt={YYYY-MM-DD}/segment=offer_status/part-*.parquet
  
  Schema:
    - id:             str (offer ID)
    - source:         str ("FT" | "WTTJ")
    - status:         str ("published" | "unpublished" | "reappeared")
    - published_at:   datetime (first seen date)
    - unpublished_at: datetime (date disappeared, NULL if still published)
    - reappeared_at:  datetime (date reappeared after unpublication, NULL if never)
    - first_seen_dt:  str (YYYY-MM-DD, date of first appearance)
    - last_seen_dt:   str (YYYY-MM-DD, date of last appearance)

KEY BENEFITS
============

1. **Immutability & Reproducibility**
   - Merged dataset remains pure and immutable
   - Status is versioned by date (one status row per date per offer)
   - Can replay history: combine all status records to reconstruct any past state

2. **Efficient Incremental Updates**
   - Only compute status changes for NEW merged_dt (not all historical dates)
   - Append-only: new dt partitions added, never retroactively modified
   - Daily cost ∝ size of latest dataset, not cumulative history

3. **Native KPI Calculation**
   - Query directly: "SELECT COUNT(*) WHERE status='unpublished' AND unpublished_at BETWEEN X AND Y"
   - Compute churn rate: unpublications per source per date
   - Reappearance rate: reappeared offers / disappeared offers
   - Cohort analysis: first_seen_dt → retention by source over time
   - No need to query merged dataset at all

4. **Separation of Concerns**
   - Status calculation decoupled from offer merging
   - Can be recomputed/debugged independently
   - Easier to add new status rules without modifying merge logic

OPERATIONAL MODES
=================

Two modes control how status history is computed:

A. **INCREMENTAL MODE** (Default, Fast)
   ────────────────────────────────────
   Use case: Daily pipeline runs
   Algorithm:
     1. Load merged_dt (latest)
     2. Load status_history from previous_dt (if exists)
     3. Compare IDs:
        - Offers in previous but not in latest → mark unpublished_at = merged_dt
        - Offers in latest but not in previous → mark published_at = merged_dt
        - Offers in both → update last_seen_dt = merged_dt
     4. Append only new records for merged_dt to status_history/dt={merged_dt}/
   
   Cost: O(n_latest_offers) per run
   Data growth: Linear in number of new dates
   
   Example query:
     SELECT id, source, unpublished_at 
     FROM status_history/dt=2026-03-08 
     WHERE status='unpublished' AND unpublished_at='2026-03-08'

B. **GLOBAL RECALCULATION MODE** (Expensive, Complete)
   ───────────────────────────────────────────────────
   Use case: Fixing bugs, debugging, validation after data corrections
   Algorithm:
     1. List ALL merged datasets (from oldest to newest)
     2. Iterate chronologically, tracking offer presence timeline
     3. For each offer ID, compute:
        - first_seen_dt = earliest date with presence
        - last_seen_dt = latest date with presence
        - unpublished_at = first date with absence after presence
        - reappeared_at = first date with presence after unpublication
     4. Completely overwrite status_history/dt=*/ with recomputed records
   
   Cost: O(n_all_offers * n_all_dates) - expensive but validates consistency
   Data: Replace all status history partitions
   
   Example: Identify offers that disappeared and never reappeared
     SELECT id, source, unpublished_at
     FROM status_history (global recalc result)
     WHERE reappeared_at IS NULL AND status='unpublished'

INTEGRATION WITH PIPELINE
==========================

Current pipeline structure:
  bronze/{source}/dt=.../run_id=.../{segment}/ 
    ↓ normalize_*.py (per-source)
  silver/merged/dt=.../run_id=.../merged_dataset.parquet
    ↓ THIS MODULE ↓
  silver/jobmarket/status_history/dt=.../segment=offer_status/status_dataset.parquet

Daily execution (INCREMENTAL MODE):
  1. Normalize FT and WTTJ sources → silver/merged/dt={today}/
  2. Run status tracking (incremental mode)
     → Compares with silver/merged/dt={yesterday}/
     → Appends to status_history/dt={today}/
  3. KPI dashboard/reports query status_history/dt=*/ for daily metrics

Validation run (GLOBAL RECALCULATION):
  1. Enable --mode=global_recalc flag
  2. Replays all merged datasets from first to last
  3. Regenerates complete status history
  4. Compare with incremental result to catch any bugs
===================================================================================
"""

import argparse
import gc
import logging
import numpy as np
import pandas as pd
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict
from src.storage.storage import Storage, get_storage_from_env
from src.utils import storage_tools
from src.utils.log_to_db import log_to_db

logger = logging.getLogger(__name__)

def build_status_history_dataset(
    df_old: Optional[pd.DataFrame] = None,
    df_new: pd.DataFrame = None,
    current_dt: str = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Build an immutable status-history snapshot for `current_dt` from two merged datasets.

    This function implements the optimized vectorized approach (Solution 1):
    - no row-by-row Python loops (`iterrows` removed)
    - no O(n) DataFrame lookup inside loops
    - all lifecycle transitions computed with `isin`, `merge`, `np.where`, and `concat`

    Why this architecture:
    - Keeps merged dataset immutable (status lives in a dedicated dataset)
    - Scales to large daily volumes (hundreds of thousands of offers)
    - Produces deterministic, replayable snapshots per date

    Performance model:
    - Previous approach: nested row iteration with repeated lookup, close to O(n^2)
    - Current approach: hash/set style matching + vectorized transforms, closer to O(n)

    Lifecycle rules applied:
    1. Existing offer (old and new):
       - previous status="unpublished" -> status="reappeared"
       - otherwise -> status="published"
       - `last_seen_dt` updated to `current_dt`
    2. Disappeared offer (old only):
       - status="unpublished"
       - set `unpublished_at` to `current_dt` if not already unpublished
    3. New offer (new only):
       - status="published"
       - initialize first/published/last seen fields to `current_dt`

    Output schema:
      - id (str): offer ID
      - source (str): FT or WTTJ
      - status (str): published | unpublished | reappeared
      - published_at (datetime)
      - unpublished_at (datetime | NULL)
      - reappeared_at (datetime | NULL)
      - first_seen_dt (str: YYYY-MM-DD)
      - last_seen_dt (str: YYYY-MM-DD)

    Args:
        df_old: Previous merged snapshot; optional for initial run.
        df_new: Current merged snapshot; must include at least `id`, `source`.
        current_dt: Processing date (YYYY-MM-DD). Defaults to UTC today.

    Returns:
        Tuple[pd.DataFrame, dict]:
        - status_df: one status snapshot DataFrame for `current_dt`
        - stats_dict: per-source counts for published/unpublished/reappeared
    """

    if df_new is None or len(df_new) == 0:
        raise ValueError("df_new must be provided and non-empty")

    required_new_cols = {"id", "source"}
    missing_new_cols = required_new_cols.difference(df_new.columns)
    if missing_new_cols:
        raise ValueError(f"df_new is missing required columns: {sorted(missing_new_cols)}")

    if current_dt is None:
        current_dt = datetime.now().strftime("%Y-%m-%d")

    current_timestamp = datetime.strptime(current_dt, "%Y-%m-%d")

    output_columns = [
        "id",
        "source",
        "status",
        "published_at",
        "unpublished_at",
        "reappeared_at",
        "first_seen_dt",
        "last_seen_dt",
    ]

    # =============================
    # 1. Initial run - no previous dataset
    # =============================
    if df_old is None or len(df_old) == 0:
        logger.info("📌 Initial run: creating status records for all offers (all 'published')")

        # Vectorized initialization for all offers on first run.
        df_status = df_new[["id", "source"]].copy()
        df_status["status"] = "published"
        df_status["published_at"] = current_timestamp
        df_status["unpublished_at"] = None
        df_status["reappeared_at"] = None
        df_status["first_seen_dt"] = current_dt
        df_status["last_seen_dt"] = current_dt

        # Compute stats by source
        stats = {
            source: {
                "published": len(df_status[df_status["source"] == source]),
                "unpublished": 0,
                "reappeared": 0
            }
            for source in df_status["source"].unique()
        }

        logger.info(f"   Created {len(df_status):,} initial status records")
        return df_status[output_columns], stats

    # =============================
    # 2. Subsequent runs - compare datasets
    # =============================
    logger.info(f"📊 Comparing datasets: {len(df_old):,} previous vs {len(df_new):,} current")
    required_old_cols = {"id", "source"}
    missing_old_cols = required_old_cols.difference(df_old.columns)
    if missing_old_cols:
        raise ValueError(f"df_old is missing required columns: {sorted(missing_old_cols)}")

    # Create composite keys for comparison (source + id)
    df_old_keyed = df_old.copy()
    df_new_keyed = df_new.copy()

    # Ensure optional lifecycle columns exist to keep logic robust with legacy merged files.
    if "status" not in df_old_keyed.columns:
        df_old_keyed["status"] = "published"
    else:
        df_old_keyed["status"] = df_old_keyed["status"].fillna("published")

    for column_name, default_value in [
        ("published_at", current_timestamp),
        ("unpublished_at", None),
        ("reappeared_at", None),
        ("first_seen_dt", current_dt),
        ("last_seen_dt", current_dt),
    ]:
        if column_name not in df_old_keyed.columns:
            df_old_keyed[column_name] = default_value
        else:
            df_old_keyed[column_name] = df_old_keyed[column_name].fillna(default_value)

    df_old_keyed["composite_key"] = df_old_keyed["source"] + "|" + df_old_keyed["id"].astype(str)
    df_new_keyed["composite_key"] = df_new_keyed["source"] + "|" + df_new_keyed["id"].astype(str)

    old_key_series = df_old_keyed["composite_key"]
    new_key_series = df_new_keyed["composite_key"]

    existing_mask = old_key_series.isin(new_key_series)
    disappeared_mask = ~existing_mask
    new_mask = ~new_key_series.isin(old_key_series)

    existing_count = int(existing_mask.sum())
    disappeared_count = int(disappeared_mask.sum())
    new_count = int(new_mask.sum())

    logger.info(f"   Existing offers: {existing_count:,}")
    logger.info(f"   Disappeared offers: {disappeared_count:,}")
    logger.info(f"   New/reappeared offers: {new_count:,}")

    # Existing (old and new): published, or reappeared if previously unpublished.
    existing_df = df_old_keyed.loc[existing_mask, [
        "id",
        "source",
        "status",
        "published_at",
        "unpublished_at",
        "first_seen_dt",
    ]].copy()
    was_unpublished_mask = existing_df["status"].eq("unpublished")
    existing_df["status"] = np.where(was_unpublished_mask, "reappeared", "published")
    existing_df["reappeared_at"] = np.where(was_unpublished_mask, current_timestamp, None)
    existing_df["unpublished_at"] = np.where(was_unpublished_mask, existing_df["unpublished_at"], None)
    existing_df["last_seen_dt"] = current_dt

    # Disappeared (old only): mark as unpublished, keep previous unpublished_at when already unpublished.
    disappeared_df = df_old_keyed.loc[disappeared_mask, [
        "id",
        "source",
        "status",
        "published_at",
        "unpublished_at",
        "reappeared_at",
        "first_seen_dt",
        "last_seen_dt",
    ]].copy()
    already_unpublished_mask = disappeared_df["status"].eq("unpublished")
    disappeared_df["status"] = "unpublished"
    disappeared_df["unpublished_at"] = np.where(
        already_unpublished_mask,
        disappeared_df["unpublished_at"],
        current_timestamp,
    )
    disappeared_df["last_seen_dt"] = current_dt

    # New (new only): initialize lifecycle fields.
    new_df = df_new_keyed.loc[new_mask, ["id", "source"]].copy()
    new_df["status"] = "published"
    new_df["published_at"] = current_timestamp
    new_df["unpublished_at"] = None
    new_df["reappeared_at"] = None
    new_df["first_seen_dt"] = current_dt
    new_df["last_seen_dt"] = current_dt

    df_status = pd.concat(
        [
            existing_df[output_columns],
            disappeared_df[output_columns],
            new_df[output_columns],
        ],
        ignore_index=True,
        sort=False,
    )

    # =============================
    # Compute comprehensive stats by source
    # =============================
    stats = {}
    if len(df_status) > 0:
        stats_df = (
            df_status.groupby(["source", "status"]).size().unstack(fill_value=0)
        )
        for source, row in stats_df.iterrows():
            stats[source] = {
                "published": int(row.get("published", 0)),
                "unpublished": int(row.get("unpublished", 0)),
                "reappeared": int(row.get("reappeared", 0)),
            }

    # =============================
    # Log summary
    # =============================
    logger.info("✅ Status history computed:")
    for source, counts in stats.items():
        logger.info(f"   {source}:")
        logger.info(f"      - Published: {counts['published']:,}")
        logger.info(f"      - Unpublished: {counts['unpublished']:,}")
        logger.info(f"      - Reappeared: {counts['reappeared']:,}")
    
    return df_status, stats



# =============================
# Storage I/O Functions
# =============================

def get_latest_parquet_files(storage: Storage, prefix: str = "", limit: Optional[int] = 2) -> List[str]:
    """
    Get parquet files from storage, sorted by merged_dt in filename (ascending).
    
    Args:
        storage: Storage instance
        prefix: Prefix to filter files (default: "" for all files)
        limit: Maximum number of files to return (default: 2 for comparison, None for all files)
    
    Returns:
        List of parquet file keys sorted by merged_dt ascending.
    """
    try:
        # List all objects in the storage
        all_objects = list(storage.list_keys(prefix))

        # Filter parquet files only
        parquet_files = [obj for obj in all_objects if obj.endswith('.parquet')]

        # Sort by merged_dt=YYYY-MM-DD from the new naming convention:
        # merged_dt=2026-03-07_ft_dt=2026-03-07_wttj_dt=2026-03-07.parquet
        # Fallback to first generic dt=... for legacy files.
        def _dt_sort_key(filename: str) -> datetime:
            merged_match = re.search(r"merged_dt=(\d{4}-\d{2}-\d{2})", filename)
            dt_str = merged_match.group(1) if merged_match else None

            if dt_str is None:
                generic_match = re.search(r"dt=(\d{4}-\d{2}-\d{2})", filename)
                dt_str = generic_match.group(1) if generic_match else None

            if dt_str is None:
                return datetime.min

            try:
                return datetime.strptime(dt_str, "%Y-%m-%d")
            except ValueError:
                return datetime.min

        parquet_files.sort(key=_dt_sort_key)
        
        logger.info(f"📋 Found {len(parquet_files)} parquet file(s) in storage")
        
        # Return the latest N files while preserving ascending order.
        # If limit is None, return all files
        if limit is None:
            return parquet_files
        else:
            return parquet_files[-limit:]
        
    except Exception as e:
        logger.error(f"❌ Error listing parquet files: {e}")
        return []


def run_status_tracking(
    storage_silver_merged: Optional[Storage] = None,
    storage_silver_status: Optional[Storage] = None,
    mode: str = "incremental",
    output_prefix: Optional[str] = None,
) -> dict:
    """
    Main function to track offer status across consecutive datasets.
    
    INCREMENTAL MODE (default):
    ────────────────────────────
    Process:
    1. Load the two most recent merged datasets from storage
    2. Compare them to identify disappeared/new/reappeared offers
    3. Build status history dataset for this merged_dt
    4. Save to separate status_history partition (dt={merged_dt})
    
    Cost: O(n_latest_offers) per run
    Data: Append-only (one dt partition per run)
    
    Arguments:
        storage_silver_merged: Storage for merged datasets (default: env)
        storage_silver_status: Storage for status history (default: env silver/status_history)
        mode: "incremental" (default) or "global_recalc"
        output_prefix: Prefix for output file (default: offer_status.parquet)
    
    Returns:
        Dict with success status, stats, and output key
    """
    logger.info("=" * 80)
    logger.info("🔄 STATUS TRACKING - OFFER LIFECYCLE MONITORING")
    logger.info(f"📋 Mode: {mode.upper()}")
    logger.info("=" * 80)
    
    # Generate unique job_id for this run
    job_id = f"status-tracking-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    # Log start to DB
    try:
        log_to_db(
            endpoint="calculate_offer_status",
            level="INFO",
            message=f"Start status tracking job (mode={mode}): {job_id}",
            task_id=job_id,
            status="RUNNING",
            mode=mode
        )
    except Exception as e:
        logger.warning(f"[calculate_offer_status] log_to_db start failed: {e}")
    
    # Initialize storage if not provided
    if storage_silver_merged is None:
        storage_silver_merged = get_storage_from_env("silver", "merged")
        logger.info(f"📂 Storage (merged): silver/merged")
    
    if storage_silver_status is None:
        storage_silver_status = get_storage_from_env("silver", "status_history")
        logger.info(f"📂 Storage (status): silver/status_history")
    
    # =============================
    # INCREMENTAL MODE
    # =============================
    if mode == "incremental":
        return _run_incremental_status_update(
            storage_merged=storage_silver_merged,
            storage_status=storage_silver_status,
            job_id=job_id,
            start_time=start_time,
            output_prefix=output_prefix
        )
    
    # =============================
    # GLOBAL RECALCULATION MODE
    # =============================
    elif mode == "global_recalc":
        return _run_global_status_recalculation(
            storage_merged=storage_silver_merged,
            storage_status=storage_silver_status,
            job_id=job_id,
            start_time=start_time,
            output_prefix=output_prefix
        )
    
    else:
        error_msg = f"Unknown mode: {mode}. Use 'incremental' or 'global_recalc'"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "status_stats": {}
        }


def _run_incremental_status_update(
    storage_merged: Storage,
    storage_status: Storage,
    job_id: str,
    start_time: float,
    output_prefix: Optional[str] = None
) -> dict:
    """
    INCREMENTAL MODE: Compare two latest merged datasets, update status for current dt only.
    
    Efficient daily operation: only O(n_latest_offers) cost.
    Appends status records to status_history/dt={current_dt}/
    """
    logger.info("")
    logger.info("🔍 INCREMENTAL MODE: Searching for recent merged datasets...")
    
    parquet_files = get_latest_parquet_files(storage_merged, prefix="", limit=2)
    
    if len(parquet_files) == 0:
        logger.error("❌ No parquet files found in storage")
        return {
            "success": False,
            "message": "No datasets found in storage",
            "status_stats": {}
        }
    
    if len(parquet_files) == 1:
        logger.info("ℹ️  Only one dataset found - skipping status comparison (first run)")
        logger.info(f"   Dataset: {parquet_files[0]}")
        return {
            "success": True,
            "message": "Only one dataset available - status tracking requires at least 2 consecutive datasets",
            "current_dataset": parquet_files[0],
            "status_stats": {}
        }
    
    # Load the two most recent datasets
    current_file = parquet_files[-1]  # Most recent
    previous_file = parquet_files[-2]  # Previous
    
    logger.info(f"📥 Loading datasets for comparison:")
    logger.info(f"   Current (new):  {current_file}")
    logger.info(f"   Previous (old): {previous_file}")
    
    df_current = storage_tools.load_parquet_dataset(storage_merged, current_file)
    df_previous = storage_tools.load_parquet_dataset(storage_merged, previous_file)
    
    if df_current is None:
        logger.error(f"❌ Failed to load current dataset: {current_file}")
        return {
            "success": False,
            "message": f"Failed to load current dataset: {current_file}",
            "status_stats": {}
        }
    
    if df_previous is None:
        logger.warning(f"⚠️ Failed to load previous dataset: {previous_file}")
        logger.info("   Proceeding without comparison")
        df_previous = None
    
    # Extract dataset date from current_file name
    merged_dt_match = re.search(r"merged_dt=(\d{4}-\d{2}-\d{2})", current_file)
    if merged_dt_match:
        current_dt = merged_dt_match.group(1)
        logger.info(f"   Using dataset date as reference: {current_dt}")
    else:
        current_dt = datetime.now().strftime("%Y-%m-%d")
        logger.warning(f"   Could not extract merged_dt, using current date: {current_dt}")
    
    logger.info("")
    logger.info("🔄 Building status history dataset...")
    
    # Build status history (as separate dataset)
    df_status, status_stats = build_status_history_dataset(
        df_old=df_previous,
        df_new=df_current,
        current_dt=current_dt
    )
    
    # Save status history to separate partition
    logger.info("")
    logger.info("💾 Saving status history dataset...")
    
    # Partition by merged_dt
    dt_partition = current_dt
    run_prefix = f"dt={dt_partition}/segment=offer_status/"
    run_json_key = f"dt={dt_partition}/run=offer_status_{job_id}/"
    
    output_key = f"{run_prefix}offer_status.parquet"
    if output_prefix:
        output_key = f"{run_prefix}{output_prefix}.parquet"
    
    success = storage_tools.save_parquet_dataset(storage_status, df_status, output_key)
    
    # Create run.json metadata
    if success:
        run_metadata = {
            "job_id": job_id,
            "mode": "incremental",
            "timestamp_start": datetime.fromtimestamp(start_time).isoformat(),
            "timestamp_end": datetime.now().isoformat(),
            "merged_dt": current_dt,
            "current_merged_file": current_file,
            "previous_merged_file": previous_file if df_previous is not None else None,
            "total_status_records": len(df_status),
            "status_stats": status_stats,
            "output_file": output_key,
            "processing_steps": [
                "load_two_latest_merged_datasets",
                "compare_composite_keys",
                "build_status_history_dataset",
                "save_to_status_history_partition"
            ]
        }
        
        try:
            storage_status.write_json(f"{run_json_key}run.json", run_metadata)
            logger.info(f"   ℹ️ Metadata saved: {run_json_key}run.json")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to save run.json metadata: {e}")
    
    elapsed_sec = time.time() - start_time
    
    if success:
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ INCREMENTAL STATUS UPDATE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"📊 Output: {output_key}")
        logger.info(f"📈 Status records: {len(df_status):,}")
        
        try:
            total_published = sum(stats.get('published', 0) for stats in status_stats.values())
            total_unpublished = sum(stats.get('unpublished', 0) for stats in status_stats.values())
            total_reappeared = sum(stats.get('reappeared', 0) for stats in status_stats.values())
            
            log_to_db(
                endpoint="calculate_offer_status",
                level="INFO",
                message=f"Incremental status update completed: {len(df_status):,} status records, {total_unpublished:,} unpublished, {total_reappeared:,} reappeared",
                task_id=job_id,
                duration_sec=elapsed_sec,
                records_count=len(df_status),
                status="SUCCESS",
                output_key=output_key,
                published=total_published,
                unpublished=total_unpublished,
                reappeared=total_reappeared,
                mode="incremental"
            )
        except Exception as e:
            logger.warning(f"[calculate_offer_status] log_to_db success failed: {e}")
        
        return {
            "success": True,
            "message": f"Incremental status update completed: {len(df_status):,} status records",
            "mode": "incremental",
            "output_key": output_key,
            "run_json_key": f"{run_json_key}run.json",
            "merged_dt": current_dt,
            "total_status_records": len(df_status),
            "status_stats": status_stats,
            "elapsed_sec": elapsed_sec
        }
    else:
        logger.error("❌ Failed to save status history dataset")
        
        try:
            log_to_db(
                endpoint="calculate_offer_status",
                level="ERROR",
                message=f"Failed to save status history to {output_key}",
                task_id=job_id,
                duration_sec=elapsed_sec,
                status="ERROR",
                mode="incremental"
            )
        except Exception as e:
            logger.warning(f"[calculate_offer_status] log_to_db error failed: {e}")
        
        return {
            "success": False,
            "message": "Failed to save status history dataset",
            "mode": "incremental",
            "status_stats": status_stats
        }


def _run_global_status_recalculation(
    storage_merged: Storage,
    storage_status: Storage,
    job_id: str,
    start_time: float,
    output_prefix: Optional[str] = None
) -> dict:
    """
    GLOBAL RECALCULATION MODE: Replay ALL merged datasets chronologically.
    
    Expensive but comprehensive: replays entire history to verify status consistency.
    Overwrites ALL status_history/dt=*/ partitions with recomputed records.
    
    Use when:
    - Debugging inconsistencies
    - After fixing merged dataset bugs
    - Validating incremental updates are correct
    """
    logger.info("")
    logger.info("🔍 GLOBAL RECALCULATION MODE: Scanning all merged datasets...")
    
    # Get ALL merged datasets (sorted chronologically)
    all_parquet_files = get_latest_parquet_files(storage_merged, prefix="", limit=None)
    
    if len(all_parquet_files) < 2:
        logger.error("❌ Need at least 2 datasets for status calculation")
        return {
            "success": False,
            "message": "Insufficient datasets found for global recalculation (need ≥2)",
            "status_stats": {}
        }
    
    logger.info(f"📋 Found {len(all_parquet_files)} merged dataset(s) for replay")
    logger.info(f"   First: {all_parquet_files[0]}")
    logger.info(f"   Last:  {all_parquet_files[-1]}")
    logger.info("")
    
    all_stats = {}
    total_status_records = 0
    save_results = {}
    df_previous = None

    for file_idx, merged_file in enumerate(all_parquet_files, start=1):
        logger.info(f"[{file_idx:>3d}/{len(all_parquet_files):>3d}] Processing: {merged_file}")

        df_current = storage_tools.load_parquet_dataset(storage_merged, merged_file)
        if df_current is None:
            logger.warning(f"            ⚠️ Failed to load, skipping")
            continue

        logger.info(f"            ✓ Loaded {len(df_current):,} rows")

        merged_dt_match = re.search(r"merged_dt=(\d{4}-\d{2}-\d{2})", merged_file)
        current_dt = merged_dt_match.group(1) if merged_dt_match else None

        if current_dt is None:
            logger.warning(f"            ⚠️ Could not extract date from filename, skipping")
            continue

        logger.info(f"            Building status history for {current_dt}...")

        # Pass only needed columns to avoid full-DataFrame copies inside build_status_history_dataset:
        # - df_new only needs id + source
        # - df_old needs the 8 status lifecycle columns (already slimmed from previous iteration)
        df_status, stats = build_status_history_dataset(
            df_old=df_previous,
            df_new=df_current[["id", "source"]],
            current_dt=current_dt
        )

        logger.info(f"            ✓ {len(df_status):,} status records computed")

        # Save immediately and free df_status — avoids accumulating all DataFrames in memory
        run_prefix = f"dt={current_dt}/segment=offer_status/"
        output_key = f"{run_prefix}offer_status.parquet"
        if output_prefix:
            output_key = f"{run_prefix}{output_prefix}.parquet"

        success = storage_tools.save_parquet_dataset(storage_status, df_status, output_key)
        save_results[current_dt] = {
            "success": success,
            "output_key": output_key,
            "records": len(df_status),
        }
        if success:
            total_status_records += len(df_status)
            logger.info(f"[{file_idx:>3d}/{len(all_parquet_files):>3d}] ✅ {current_dt}: {len(df_status):,} records → {output_key}")
        else:
            logger.error(f"[{file_idx:>3d}/{len(all_parquet_files):>3d}] ❌ {current_dt}: Failed to save")

        all_stats[current_dt] = stats

        # Slim df_current to only the columns needed as df_old in the next iteration, then free df_status
        _STATUS_COLS = ["id", "source", "status", "published_at", "unpublished_at", "reappeared_at", "first_seen_dt", "last_seen_dt"]
        df_previous = df_status[[c for c in _STATUS_COLS if c in df_status.columns]].copy()
        del df_status
        del df_current
        gc.collect()

    if len(save_results) == 0:
        logger.error("❌ No valid merged datasets found for global recalculation")
        return {
            "success": False,
            "message": "No valid datasets for global recalculation",
            "status_stats": {}
        }
    
    elapsed_sec = time.time() - start_time
    
    successful_saves = sum(1 for r in save_results.values() if r["success"])
    
    if successful_saves == len(save_results):
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ GLOBAL STATUS RECALCULATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"📊 Dates processed: {successful_saves}")
        logger.info(f"📈 Total status records: {total_status_records:,}")
        logger.info(f"⏱️  Elapsed time: {elapsed_sec:.1f}s")
        
        try:
            log_to_db(
                endpoint="calculate_offer_status",
                level="INFO",
                message=f"Global status recalculation completed: {total_status_records:,} records across {successful_saves} dates",
                task_id=job_id,
                duration_sec=elapsed_sec,
                records_count=total_status_records,
                status="SUCCESS",
                dates_processed=successful_saves,
                mode="global_recalc"
            )
        except Exception as e:
            logger.warning(f"[calculate_offer_status] log_to_db success failed: {e}")
        
        return {
            "success": True,
            "message": f"Global recalculation completed: {total_status_records:,} records across {successful_saves} dates",
            "mode": "global_recalc",
            "dates_processed": successful_saves,
            "total_status_records": total_status_records,
            "status_stats": all_stats,
            "save_results": save_results,
            "elapsed_sec": elapsed_sec
        }
    else:
        logger.warning(f"⚠️  Partial completion: {successful_saves}/{len(save_results)} dates saved successfully")
        
        try:
            log_to_db(
                endpoint="calculate_offer_status",
                level="WARNING",
                message=f"Global recalculation partially completed: {successful_saves}/{len(save_results)} dates",
                task_id=job_id,
                duration_sec=elapsed_sec,
                status="PARTIAL",
                mode="global_recalc"
            )
        except Exception as e:
            logger.warning(f"[calculate_offer_status] log_to_db warning failed: {e}")
        
        return {
            "success": False,
            "message": f"Global recalculation partial: {successful_saves}/{len(save_results)} dates saved",
            "mode": "global_recalc",
            "dates_processed": successful_saves,
            "total_status_records": total_status_records,
            "status_stats": all_stats,
            "save_results": save_results,
            "elapsed_sec": elapsed_sec
        }



def main():
    """Entry point for CLI execution"""
    parser = argparse.ArgumentParser(
        description="Track job offer status by comparing consecutive merged datasets"
    )
    
    # Support both --mode (preferred) and --compute-status (legacy) for backwards compatibility
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        choices=["incremental", "global_recalc"],
        help="Tracking mode: 'incremental' (default, fast daily) or 'global_recalc' (slow, validates all history)"
    )
    mode_group.add_argument(
        "--compute-status",
        choices=["incremental", "global_recalc"],
        help="[DEPRECATED] Use --mode instead. Tracking mode."
    )
    
    parser.add_argument(
        "--output-prefix",
        help="Prefix for output file (default: offer_status.parquet)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Determine mode: prefer --mode, fallback to --compute-status
    if args.mode:
        mode = args.mode
    elif args.compute_status:
        mode = args.compute_status
    else:
        mode = "incremental"  # default
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    # Run status tracking
    result = run_status_tracking(mode=mode, output_prefix=args.output_prefix)
    
    # Exit with appropriate code
    if result["success"]:
        logger.info("✅ Status tracking completed successfully")
        exit(0)
    else:
        logger.error(f"❌ Status tracking failed: {result['message']}")
        exit(1)


if __name__ == "__main__":
    main()
