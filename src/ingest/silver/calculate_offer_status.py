# ==========================
# Status Tracking Module
# ==========================
# Track job offer lifecycle: published -> unpublished detection
# Compares two consecutive merged datasets to identify disappeared offers
# Uses composite key (source + id) for uniqueness

import argparse
import logging
import pandas as pd
import re
import time
import uuid
from datetime import datetime
from typing import Tuple, Optional, List
from src.storage.storage import Storage, get_storage_from_env
from src.utils import storage_tools
from src.utils.log_to_db import log_to_db

logger = logging.getLogger(__name__)


def calculate_offer_status(
    df_old: Optional[pd.DataFrame] = None,
    df_new: pd.DataFrame = None,
    current_timestamp: datetime = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Calculate and update offer status based on dataset comparison.
    
    Strategy:
    1. First run (no old dataset): All offers marked as 'published'
    2. Subsequent runs: Compare with old dataset
       - Disappeared offers (in old but not in new) → status='unpublished' + unpublished_at
       - New/existing offers → status='published' (or update if status was previously 'unpublished')
    
    Args:
        df_old: Previous merged dataset (None for initial run)
        df_new: Current merged dataset
        current_timestamp: Timestamp for unpublished_at field (defaults to now)
    
    Returns:
        Tuple[df_updated, stats_dict] with:
        - df_updated: DataFrame with updated status and unpublished_at fields
        - stats_dict: Dictionary with counts by source (published, unpublished, appeared)
    """
    
    if current_timestamp is None:
        current_timestamp = datetime.utcnow()
    
    # =============================
    # CASE 1: Initial run - no old dataset (no comparison possible)
    # =============================
    if df_old is None or len(df_old) == 0:
        logger.info("📌 Initial run: no previous dataset for comparison, skipping status calculation")
        
        # Compute stats by source (informational only)
        stats = {
            source: {
                "published": 0,
                "unpublished": 0,
                "appeared": count
            }
            for source, count in df_new["source"].value_counts().items()
        }
        
        return df_new, stats
    
    # =============================
    # CASE 2: Subsequent runs - compare datasets
    # =============================
    logger.info(f"📊 Comparing datasets: {len(df_old):,} old vs {len(df_new):,} new")
    
    # Ensure both dataframes have status columns
    if "status" not in df_new.columns:
        df_new["status"] = ""
    if "unpublished_at" not in df_new.columns:
        df_new["unpublished_at"] = None
    
    # Create composite keys for comparison (source + id)
    df_old_keyed = df_old.copy()
    df_new_keyed = df_new.copy()
    
    df_old_keyed["composite_key"] = df_old_keyed["source"] + "|" + df_old_keyed["id"].astype(str)
    df_new_keyed["composite_key"] = df_new_keyed["source"] + "|" + df_new_keyed["id"].astype(str)
    
    old_keys = set(df_old_keyed["composite_key"])
    new_keys = set(df_new_keyed["composite_key"])
    
    # Find disappeared offers (in old but not in new)
    disappeared_keys = old_keys - new_keys
    logger.info(f"   Disappeared offers: {len(disappeared_keys):,}")
    
    # Find new/reappeared offers (in new but not in old)
    new_keys_list = new_keys - old_keys
    logger.info(f"   New/reappeared offers: {len(new_keys_list):,}")

    # Keep source-level key sets from the current dataset before enrichment with disappeared rows.
    source_new_detected_keys = {
        source: set(group["composite_key"])
        for source, group in df_new_keyed.groupby("source")
    }
    
    # =============================
    # Update statuses
    # =============================
    
    # 1. Initialize status columns if missing (don't overwrite existing values)
    # df_new may already contain unpublished offers from previous iterations
    if "status" not in df_new_keyed.columns or df_new_keyed["status"].isna().all():
        df_new_keyed["status"] = "published"
    else:
        # Fill missing status values with 'published' (new offers detected in this run)
        df_new_keyed["status"] = df_new_keyed["status"].fillna("published")
    
    if "unpublished_at" not in df_new_keyed.columns:
        df_new_keyed["unpublished_at"] = None
    
    # 2. Re-inject disappeared offers from old dataset as unpublished in current output
    if disappeared_keys:
        disappeared_rows = df_old_keyed[df_old_keyed["composite_key"].isin(disappeared_keys)].copy()
        disappeared_rows.loc[:, "status"] = "unpublished"
        disappeared_rows.loc[:, "unpublished_at"] = current_timestamp

        df_new_keyed = pd.concat([df_new_keyed, disappeared_rows], ignore_index=True, sort=False)

        logger.info(
            f"   Processing disappeared offers: {len(disappeared_keys):,}/{len(disappeared_keys):,} (100.0%)"
        )
        logger.info(f"   Re-injected as unpublished: {len(disappeared_rows):,}")
    
    # 3. For offers appearing again (previously unpublished, now back)
    # Update their status to 'published'
    #  Not support  : need to track historical status in .
    #appeared_rows = df_new_keyed[df_new_keyed["composite_key"].isin(new_keys_list)]
    #for key in new_keys_list:
    #    if key in old_keys:  # Was it in old dataset?
    #        old_version = df_old_keyed[df_old_keyed["composite_key"] == key]
    #        if len(old_version) > 0 and old_version.iloc[0]["status"] == "unpublished":
    #            logger.info(f"   Offer reappeared: {key}")
    
    # =============================
    # Compute comprehensive stats by source
    # =============================
    stats = {}
    
    all_sources = sorted(set(df_old_keyed["source"]).union(set(df_new_keyed["source"])))

    for source in all_sources:
        source_new = df_new_keyed[df_new_keyed["source"] == source]
        source_old = df_old_keyed[df_old_keyed["source"] == source]
        source_detected_keys = source_new_detected_keys.get(source, set())
        
        # Count by status in output dataset (current + re-injected unpublished)
        published_count = len(source_new[source_new["status"] == "published"])
        unpublished_count = len(source_new[source_new["status"] == "unpublished"])
        
        # Count disappeared from this source
        source_old_keys = set(source_old["composite_key"])
        disappeared_from_source = len(source_old_keys - source_detected_keys)
        
        # Count newly appeared in this source
        appeared_in_source = len(source_detected_keys - source_old_keys)
        
        stats[source] = {
            "published": published_count,
            "unpublished": unpublished_count,
            "appeared": appeared_in_source,
            "total": len(source_new)
        }
    
    # =============================
    # Log summary
    # =============================
    logger.info("✅ Status update complete:")
    for source, counts in stats.items():
        logger.info(f"   {source}:")
        logger.info(f"      - Published: {counts['published']:,}")
        logger.info(f"      - Disappeared: {counts['unpublished']:,}")
        logger.info(f"      - Newly appeared: {counts['appeared']:,}")
        logger.info(f"      - Total in dataset: {counts['total']:,}")
    
    # Clean up temporary column
    df_new_keyed = df_new_keyed.drop(columns=["composite_key"])
    
    return df_new_keyed, stats


# =============================
# Storage I/O Functions
# =============================

def get_latest_parquet_files(storage: Storage, prefix: str = "", limit: int = 2) -> List[str]:
    """
    Get parquet files from storage, sorted by merged_dt in filename (ascending).
    
    Args:
        storage: Storage instance
        prefix: Prefix to filter files (default: "" for all files)
        limit: Maximum number of files to return (default: 2 for comparison)
    
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
        return parquet_files[-limit:]
        
    except Exception as e:
        logger.error(f"❌ Error listing parquet files: {e}")
        return []


def run_status_tracking(
    storage_silver_merged: Optional[Storage] = None,
    output_prefix: Optional[str] = None,
) -> dict:
    """
    Main function to track offer status across consecutive datasets.
    
    Process:
    1. Load the two most recent merged datasets from storage
    2. Compare them to identify disappeared offers
    3. Update status and unpublished_at fields
    4. Save the updated dataset back to storage
    
    Args:
        storage: Storage instance (default: get from env for silver/merged)
        output_prefix: Prefix for output file (default: same as input)
    
    Returns:
        Dict with success status, stats, and output key
    """
    logger.info("=" * 80)
    logger.info("🔄 STATUS TRACKING - OFFER LIFECYCLE MONITORING")
    logger.info("=" * 80)
    
    # Generate unique job_id for this run
    job_id = f"status-tracking-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    # Log start to DB
    try:
        log_to_db(
            endpoint="calculate_offer_status",
            level="INFO",
            message=f"Start status tracking job: {job_id}",
            task_id=job_id,
            status="RUNNING"
        )
    except Exception as e:
        logger.warning(f"[calculate_offer_status] log_to_db start failed: {e}")
    
    # Initialize storage if not provided
    if storage_silver_merged is None:
        storage_silver_merged = get_storage_from_env("silver", "merged")
        logger.info(f"📂 Storage: silver/merged")
    
    # Get the two most recent parquet files
    logger.info("🔍 Searching for recent datasets...")
    parquet_files = get_latest_parquet_files(storage_silver_merged, prefix="", limit=2)
    
    if len(parquet_files) == 0:
        logger.error("❌ No parquet files found in storage")
        try:
            log_to_db(
                endpoint="calculate_offer_status",
                level="ERROR",
                message=f"No parquet files found in storage",
                task_id=job_id,
                status="ERROR"
            )
        except Exception as e:
            logger.warning(f"[calculate_offer_status] log_to_db error failed: {e}")
        return {
            "success": False,
            "message": "No datasets found in storage",
            "status_stats": {}
        }
    
    if len(parquet_files) == 1:
        logger.info("ℹ️  Only one dataset found - skipping status comparison (first run)")
        logger.info(f"   Dataset: {parquet_files[0]}")
        try:
            log_to_db(
                endpoint="calculate_offer_status",
                level="INFO",
                message=f"Only one dataset found - skipping comparison (first run)",
                task_id=job_id,
                status="SKIPPED",
                current_dataset=parquet_files[0]
            )
        except Exception as e:
            logger.warning(f"[calculate_offer_status] log_to_db skipped failed: {e}")
        return {
            "success": True,
            "message": "Only one dataset available - status tracking requires at least 2 consecutive datasets",
            "current_dataset": parquet_files[0],
            "status_stats": {}
        }
    
    # Load the two most recent datasets from an ascending list
    current_file = parquet_files[-1]  # Most recent
    previous_file = parquet_files[-2]  # Previous
    
    logger.info(f"📥 Loading datasets for comparison:")
    logger.info(f"   Current (new):  {current_file}")
    logger.info(f"   Previous (old): {previous_file}")
    
    try:
        log_to_db(
            endpoint="calculate_offer_status",
            level="INFO",
            message=f"Loading datasets: current={current_file}, previous={previous_file}",
            task_id=job_id,
            status="LOADING",
            current_file=current_file,
            previous_file=previous_file
        )
    except Exception as e:
        logger.warning(f"[calculate_offer_status] log_to_db loading failed: {e}")
    
    df_current = storage_tools.load_parquet_dataset(storage_silver_merged, current_file)
    df_previous = storage_tools.load_parquet_dataset(storage_silver_merged, previous_file)
    
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
    
    # Calculate status updates
    logger.info("")
    logger.info("🔄 Calculating status updates...")
    
    # Extract dataset date from current_file name (merged_dt=YYYY-MM-DD)
    # Use this date as reference for unpublished_at instead of execution date
    merged_dt_match = re.search(r"merged_dt=(\d{4}-\d{2}-\d{2})", current_file)
    if merged_dt_match:
        dataset_date_str = merged_dt_match.group(1)
        current_timestamp = datetime.strptime(dataset_date_str, "%Y-%m-%d")
        logger.info(f"   Using dataset date as reference: {dataset_date_str}")
    else:
        # Fallback to execution time if merged_dt not found
        current_timestamp = datetime.utcnow()
        logger.warning(f"   Could not extract merged_dt from filename, using current time")
    
    try:
        log_to_db(
            endpoint="calculate_offer_status",
            level="INFO",
            message=f"Comparing datasets: {len(df_current):,} current vs {len(df_previous) if df_previous is not None else 0:,} previous",
            task_id=job_id,
            status="COMPARING",
            records_count=len(df_current)
        )
    except Exception as e:
        logger.warning(f"[calculate_offer_status] log_to_db comparing failed: {e}")
    
    df_updated, status_stats = calculate_offer_status(
        df_old=df_previous,
        df_new=df_current,
        current_timestamp=current_timestamp
    )
    
    # Save updated dataset
    logger.info("")
    logger.info("💾 Saving updated dataset...")
    
    # Generate output directory structure with dt= and run_id= partitions
    dt_partition = datetime.now().strftime("%Y-%m-%d")
    run_prefix = f"dt={dt_partition}/run_id={job_id}/"
    run_json_key = f"{run_prefix}run.json"
    
    # Output key for parquet file
    output_key = f"{run_prefix}merged_dataset_with_status.parquet"
    
    if output_prefix:
        output_key = f"{run_prefix}{output_prefix}.parquet"
    
    success = storage_tools.save_parquet_dataset(storage_silver_merged, df_updated, output_key)
    
    # Create run.json metadata file
    if success:
        run_metadata = {
            "job_id": job_id,
            "job_type": "status_tracking",
            "timestamp_start": datetime.fromtimestamp(start_time).isoformat(),
            "timestamp_end": datetime.now().isoformat(),
            "dt": dt_partition,
            "current_dataset": current_file,
            "previous_dataset": previous_file if df_previous is not None else None,
            "total_offers": len(df_updated),
            "status_stats": status_stats,
            "output_file": output_key,
            "processing_steps": [
                "load_datasets",
                "compare_composite_keys",
                "calculate_status_changes",
                "save_updated_dataset"
            ]
        }
        
        try:
            storage_silver_merged.write_json(run_json_key, run_metadata)
            logger.info(f"   ℹ️ Metadata saved: {run_json_key}")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to save run.json metadata: {e}")
    
    elapsed_sec = time.time() - start_time
    
    if success:
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ STATUS TRACKING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"📊 Output: {output_key}")
        logger.info(f"📝 Metadata: {run_json_key}")
        logger.info(f"📈 Total offers: {len(df_updated):,}")
        
        # Log success to DB
        try:
            # Compute total stats across all sources
            total_published = sum(stats.get('published', 0) for stats in status_stats.values())
            total_unpublished = sum(stats.get('unpublished', 0) for stats in status_stats.values())
            total_appeared = sum(stats.get('appeared', 0) for stats in status_stats.values())
            
            log_to_db(
                endpoint="calculate_offer_status",
                level="INFO",
                message=f"Status tracking completed: {len(df_updated):,} offers processed, {total_unpublished:,} disappeared, {total_appeared:,} appeared",
                task_id=job_id,
                duration_sec=elapsed_sec,
                records_count=len(df_updated),
                status="SUCCESS",
                output_key=output_key,
                published=total_published,
                unpublished=total_unpublished,
                appeared=total_appeared
            )
        except Exception as e:
            logger.warning(f"[calculate_offer_status] log_to_db success failed: {e}")
        
        return {
            "success": True,
            "message": f"Status tracking completed: {len(df_updated):,} offers processed",
            "output_key": output_key,
            "run_json_key": run_json_key,
            "run_prefix": run_prefix,
            "current_dataset": current_file,
            "previous_dataset": previous_file,
            "total_offers": len(df_updated),
            "status_stats": status_stats,
            "elapsed_sec": elapsed_sec
        }
    else:
        logger.error("❌ Failed to save updated dataset")
        
        # Log failure to DB
        try:
            log_to_db(
                endpoint="calculate_offer_status",
                level="ERROR",
                message=f"Failed to save updated dataset to {output_key}",
                task_id=job_id,
                duration_sec=elapsed_sec,
                status="ERROR"
            )
        except Exception as e:
            logger.warning(f"[calculate_offer_status] log_to_db error failed: {e}")
        
        return {
            "success": False,
            "message": "Failed to save updated dataset",
            "status_stats": status_stats
        }


# =============================
# CLI Entry Point
# =============================

def main():
    """Entry point for CLI execution"""
    parser = argparse.ArgumentParser(
        description="Track job offer status by comparing consecutive merged datasets"
    )
    parser.add_argument(
        "--output-prefix",
        help="Prefix for output file (default: merged_dataset_with_status)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    # Run status tracking
    result = run_status_tracking(output_prefix=args.output_prefix)
    
    # Exit with appropriate code
    if result["success"]:
        logger.info("✅ Status tracking completed successfully")
        exit(0)
    else:
        logger.error(f"❌ Status tracking failed: {result['message']}")
        exit(1)


if __name__ == "__main__":
    main()
