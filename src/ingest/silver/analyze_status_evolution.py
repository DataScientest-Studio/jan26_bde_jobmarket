"""
Build status evolution analytics datasets, KPIs, and monitoring charts.

This module is intentionally separated from `calculate_offer_status.py`.
It provides a two-phase analytics workflow:

**Phase 1: Dataset Parquet Generation** (`run_status_evolution_parquet_generation`)
- Loads status_history and merged snapshots generated previously by `calculate_offer_status.py` for each date
- Builds complete latest-state dataset combining status + merged data
- Generates status_timeline (all snapshots concatenated with snapshot_dt)
- Writes parquets only (complete_dataset + status_timeline)
- No KPI computation, no visualization in this phase

**Phase 2: Analytics Computation & Visualization** (`run_status_evolution_analytics_computation`)
- Auto-discovers latest status_timeline parquet from Phase 1 output
- Computes daily and cumulative KPI aggregates
- Generates temporal monitoring charts (PNG files)
- Logs KPI summaries
- Independent of Phase 1 - can be run separately at any time

Data sources:
- silver/merged/*.parquet (latest merged snapshot)
- silver/status_history/dt=.../segment=offer_status/*.parquet (status snapshots)

Main outputs (written under silver/status_analytics):
- dt={analysis_dt}/segment=complete_dataset/complete_offers_with_status.parquet
- dt={analysis_dt}/segment=status_timeline/status_timeline.parquet
- dt={analysis_dt}/segment=kpis/daily_status_kpis.parquet
- dt={analysis_dt}/segment=plots/*.png
- dt={analysis_dt}/run_id={job_id}/run.json (Phase 1 only, no chart metadata)

Usage:
- Phase 1 only: python -m src.ingest.silver.analyze_status_evolution --skip-computation
- Both phases: python -m src.ingest.silver.analyze_status_evolution
- Phase 2 only: can be run independently with existing Phase 1 outputs
"""

import argparse
import io
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configure matplotlib to use non-GUI backend before any matplotlib imports
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

from src.storage.storage import Storage, get_storage_from_env
from src.utils import storage_tools
from src.utils.log_to_db import log_to_db

logger = logging.getLogger(__name__)

STATUS_FIELDS = [
    "status",
    "published_at",
    "unpublished_at",
    "reappeared_at",
    "first_seen_dt",
    "last_seen_dt",
]


def _extract_dt_from_key(key: str) -> Optional[str]:
    match = re.search(r"dt=(\d{4}-\d{2}-\d{2})", key)
    return match.group(1) if match else None


def _extract_merged_dt_from_key(key: str) -> Optional[str]:
    merged_match = re.search(r"merged_dt=(\d{4}-\d{2}-\d{2})", key)
    if merged_match:
        return merged_match.group(1)
    return _extract_dt_from_key(key)


def _list_status_snapshot_files(storage_status: Storage) -> List[Tuple[str, str]]:
    """
    List all status snapshot parquet files in storage, extract their snapshot dates from keys,
     and return sorted list of (snapshot_dt, key).

    Return format: List of tuples [(snapshot_dt, key), ...] sorted by snapshot_dt ascending.
    Only includes keys that match the expected pattern for status snapshots.

    Example :
        Input keys in storage_status:
            dt=2024-01-01/segment=offer_status/file1.parquet
            dt=2024-01-02/segment=offer_status/file2.parquet
        Output:
            [('2024-01-01', 'dt=2024-01-01/segment=offer_status/file1.parquet'),
             ('2024-01-02', 'dt=2024-01-02/segment=offer_status/file2.parquet')]
    """
    status_files = []
    for key in storage_status.list_keys(""):
        if not key.endswith(".parquet"):
            continue
        if "segment=offer_status/" not in key:
            continue
        dt = _extract_dt_from_key(key)
        if not dt:
            continue
        status_files.append((dt, key))
    # Sort by snapshot_dt ascending
    status_files.sort(key=lambda x: x[0])
    return status_files


def _list_merged_files(storage_merged: Storage) -> List[Tuple[str, str]]:
    """
    list all merged parquet files in storage_merged, extract their merged_dt from keys,
    and return sorted list of (merged_dt, key).

    Return format: List of tuples [(merged_dt, key), ...] sorted by merged_dt ascending.
    """
    merged_files = []
    for key in storage_merged.list_keys(""):
        if not key.endswith(".parquet"):
            continue
        dt = _extract_merged_dt_from_key(key)
        if not dt:
            continue
        merged_files.append((dt, key))

    merged_files.sort(key=lambda x: x[0])
    return merged_files


def _load_status_timeline(storage_status: Storage) -> Tuple[pd.DataFrame, str, str]:
    """ 
    Load all status snapshot files, concatenate into a single timeline dataframe 
    with snapshot_dt column.

    Returns:
    - timeline_df: DataFrame with all status snapshots concatenated, with snapshot_dt column
    - latest_snapshot_dt: the snapshot date of the latest status file (for reference)
    - latest_snapshot_key: the storage key of the latest status file (for reference)

    """
    status_files = _list_status_snapshot_files(storage_status)
    if not status_files:
        raise RuntimeError("No status_history parquet files found")

    logger.info("Found %s status snapshot file(s)", f"{len(status_files):,}")
    logger.info("Status range: %s -> %s", status_files[0][0], status_files[-1][0])

    # Load each status snapshot file and add "snapshot_dt" column
    # Concatenate into a timeline_df list 
    timeline_parts = []
    for idx, (dt, key) in enumerate(status_files, start=1):
        logger.info("[%s/%s] Loading status snapshot %s", idx, len(status_files), key)
        df = storage_tools.load_parquet_dataset(storage_status, key)
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df["snapshot_dt"] = dt
        timeline_parts.append(df)

    if not timeline_parts:
        raise RuntimeError("Status snapshots exist but none could be loaded")
    
    # Concatenate all snapshots into a single timeline dataframe
    timeline_df = pd.concat(timeline_parts, ignore_index=True, sort=False)
    return timeline_df, status_files[-1][0], status_files[-1][1]


def _load_latest_merged(storage_merged: Storage) -> Tuple[pd.DataFrame, str, str]:
    """
    Load the latest merged parquet file from storage_merged, extract its merged_dt,
    and return the dataframe along with metadata.

    Returns:
    - df_latest: DataFrame loaded from the latest merged parquet file
    - latest_dt: the merged_dt extracted from the file key (for reference)
    - latest_key: the storage key of the latest merged parquet file (for reference)

    Example:
    If storage_merged contains:
        merged_dt=2024-01-01/merged_offers.parquet
        merged_dt=2024-01-02/merged_offers.parquet

    """
    # Load a list of all merged parquet file in merged indexed by dt and sorted by dt ascending,
    merged_files = _list_merged_files(storage_merged)
    if not merged_files:
        raise RuntimeError("No merged parquet files found")

    # Get the latest merged file (last in the sorted list), load it, and return with metadata
    latest_dt, latest_key = merged_files[-1]
    logger.info("Loading latest merged dataset: %s", latest_key)
    df_latest = storage_tools.load_parquet_dataset(storage_merged, latest_key)
    if df_latest is None or len(df_latest) == 0:
        raise RuntimeError(f"Failed to load latest merged dataset: {latest_key}")

    return df_latest, latest_dt, latest_key


def _build_complete_latest_dataset(
    latest_status_df: pd.DataFrame,
    latest_status_dt: str,
    latest_merged_df: pd.DataFrame,
    latest_merged_dt: str,
) -> pd.DataFrame:
    """ 
        Build complete latest dataset by merging latest status snapshot with latest merged snapshot.

        This dataset represents the full state of all offers at the latest snapshot date, with status info.

        It is used as the reference dataset for the latest snapshot in the status timeline and for KPI computation.
    Returns:
        - complete_df: DataFrame with one row per offer (id, source) at the latest snapshot date,
        - containing status fields from latest_status_df and merged fields from latest_merged_df.
     The merge is done on (id, source) keys. If an offer is present in status but not in merged, it will still be included with NaNs for merged fields.
     """
    required_keys = {"id", "source"}
    missing_status = required_keys.difference(latest_status_df.columns)
    missing_merged = required_keys.difference(latest_merged_df.columns)
    if missing_status:
        raise ValueError(f"latest status dataset missing required columns: {sorted(missing_status)}")
    if missing_merged:
        raise ValueError(f"latest merged dataset missing required columns: {sorted(missing_merged)}")

    merged_payload = latest_merged_df.copy()
    for c in STATUS_FIELDS:
        if c in merged_payload.columns:
            merged_payload = merged_payload.drop(columns=[c])

    complete_df = latest_status_df.merge(
        merged_payload,
        on=["id", "source"],
        how="left",
        indicator=True,
        suffixes=("", "_merged"),
    )
    complete_df["is_present_in_latest_merged"] = complete_df["_merge"].eq("both")
    complete_df = complete_df.drop(columns=["_merge"])
    complete_df["snapshot_dt"] = latest_status_dt
    complete_df["latest_merged_dt"] = latest_merged_dt

    return complete_df


def _compute_daily_status_kpis(status_timeline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily status KPIs from the complete historical status timeline.
    KPIs computed:
    - published_count: number of offers in "published" status at each snapshot date
    - unpublished_count: number of offers in "unpublished" status at each snapshot date
    - reappeared_count: number of offers in "reappeared" status at each snapshot date
    - total_records: total number of offers in the timeline at each snapshot date
    - active_stock_count: number of offers in "published" or "reappeared" status at each snapshot date
    - new_published_events_count: number of offers that transitioned to "published" status on each snapshot date
    - unpublished_events_count: number of offers that transitioned to "unpublished" status on each snapshot date
    - reappeared_events_count: number of offers that transitioned to "reappeared" status on each snapshot date
    - churn_rate: daily churn rate calculated as unpublished_events_count / active_stock_count of the previous day
    - daily_reappearance_rate: daily reappearance rate calculated as reappeared_events_count / unpublished_events_count of the same day
    - cumulative_reappearance_rate: cumulative reappearance rate calculated as cumulative reappeared_events_count / cumulative unpublished_events_count up to that day

     The computation is done by grouping the timeline dataframe by snapshot date and source, counting statuses, and calculating event counts based on date comparisons.
     The resulting KPIs dataframe has one row per snapshot date and source, with all the computed metrics for temporal analysis and monitoring.

     Return a DataFrame with columns:
        - snapshot_dt
        - source
        - published_count
        - unpublished_count
        - reappeared_count
        - total_records
        - active_stock_count
        - new_published_events_count
        - unpublished_events_count
        - reappeared_events_count
        - churn_rate
        - daily_reappearance_rate
        - cumulative_reappearance_rate

    """

    df = status_timeline_df.copy()

    # Normalize date-like columns.
    date_normalization_plan = {
        "snapshot_dt": "snapshot_dt",
        "first_seen_dt": "first_seen_dt",
        "unpublished_at": "unpublished_at_dt",
        "reappeared_at": "reappeared_at_dt",
    }
    for source_col, target_col in date_normalization_plan.items():
        if source_col in df.columns:
            df[target_col] = pd.to_datetime(df[source_col], errors="coerce").dt.strftime("%Y-%m-%d")
        elif target_col.endswith("_dt"):
            # Keep downstream event comparisons safe even if source column is absent.
            df[target_col] = None

    # Aggregate counts by snapshot date, source, and status
    status_counts = (
        df.groupby(["snapshot_dt", "source", "status"]).size().unstack(fill_value=0)
    )

    # Set KPI columns and ensure all expected status columns are present (fill missing with 0)
    status_counts["published_count"] = status_counts.get("published", 0)
    status_counts["unpublished_count"] = status_counts.get("unpublished", 0)
    status_counts["reappeared_count"] = status_counts.get("reappeared", 0)
    # Total records is the sum of all statues
    status_counts["total_records"] = (
        status_counts["published_count"]
        + status_counts["unpublished_count"]
        + status_counts["reappeared_count"]
    )

    status_counts["active_stock_count"] = (
        status_counts["published_count"] + status_counts["reappeared_count"]
    )

    #                         published  unpublished  published_count  unpublished_count  reappeared_count
    #   snapshot_dt source                                                                              
    #   2026-03-05  FT         586768            0           586768                  0                 0
    #               WTTJ        44990            0            44990                  0                 0

    # Create event flags for transitions based on date comparisons (new published, unpublished, reappeared events)
    event_flags = pd.DataFrame(
        {
            "snapshot_dt": df["snapshot_dt"],
            "source": df["source"],
            "new_published_events_count": (df["first_seen_dt"] == df["snapshot_dt"]).astype(int),
            "unpublished_events_count": (df["unpublished_at_dt"] == df["snapshot_dt"]).astype(int),
            "reappeared_events_count": (df["reappeared_at_dt"] == df["snapshot_dt"]).astype(int),
        }
    )

    # Aggregate event counts by snapshot date and source
    event_agg = event_flags.groupby(["snapshot_dt", "source"], as_index=True).sum()

    # Join two aggregations to get a complete KPI dataset by snapshot date and source
    kpis = status_counts.join(event_agg, how="left").fillna(0)
    
    # Reset index to have snapshot_dt and source as columns instead of index for easier downstream processing
    kpis = kpis.reset_index()

    # Ensure KPI columns are of integer type
    for c in [
        "new_published_events_count",
        "unpublished_events_count",
        "reappeared_events_count",
        "total_records",
        "active_stock_count",
        "published_count",
        "unpublished_count",
        "reappeared_count",
    ]:
        kpis[c] = kpis[c].astype(int)
    # Sort by source and snapshot date to ensure correct temporal order for churn rate calculations
    kpis = kpis.sort_values(["source", "snapshot_dt"]).reset_index(drop=True)

    # Previous active stock is the active stock count of the previous 
    # snapshot date for the same source (used for churn rate calculation)
    # prev_active_stock is shifted by 1 to align with the current snapshot date, 
    # so it represents the active stock at the previous snapshot date.
    # Assumes that the timeline is sorted by source and snapshot date in ascending order
    kpis["prev_active_stock"] = kpis.groupby("source")["active_stock_count"].shift(1)
    
    # Calculate churn rate as unpublished_events_count / prev_active_stock, 
    # handling division by zero and NaN cases
    kpis["churn_rate"] = np.where(
        kpis["prev_active_stock"] > 0,
        kpis["unpublished_events_count"] / kpis["prev_active_stock"],
        np.nan,
    )
    
    # Calculate daily reappearance rate as reappeared_events_count / unpublished_events_count,
    # handling division by zero and NaN cases
    kpis["daily_reappearance_rate"] = np.where(
        kpis["unpublished_events_count"] > 0,
        kpis["reappeared_events_count"] / kpis["unpublished_events_count"],
        np.nan,
    )

    # Cumulative unpublished events and cumulative reappeared events by source, 
    # used for cumulative reappearance rate calculation
    kpis["cum_unpublished_events"] = kpis.groupby("source")["unpublished_events_count"].cumsum()
    kpis["cum_reappeared_events"] = kpis.groupby("source")["reappeared_events_count"].cumsum()
    kpis["cumulative_reappearance_rate"] = np.where(
        kpis["cum_unpublished_events"] > 0,
        kpis["cum_reappeared_events"] / kpis["cum_unpublished_events"],
        np.nan,
    )

    return kpis


def _log_generated_outputs_kpis(
    storage_analytics: Storage,
    complete_key: str,
    kpis_key: str,
    analysis_dt: str,
) -> None:
    """
    Read generated analytics parquet outputs and log KPI summaries by state.

    This post-generation read-back serves two goals:
    1. Validation: ensure written parquet files are immediately readable.
    2. Observability: print KPI summaries for published/unpublished/reappeared states.
    """
    logger.info("")
    logger.info("Reading generated parquet outputs for KPI display...")

    complete_df = storage_tools.load_parquet_dataset(storage_analytics, complete_key)
    kpis_df = storage_tools.load_parquet_dataset(storage_analytics, kpis_key)

    if complete_df is None or len(complete_df) == 0:
        logger.warning("Generated complete dataset is empty or unreadable")
        return

    if "status" not in complete_df.columns:
        logger.warning("Generated complete dataset has no 'status' column")
        return

    logger.info("KPI - state counts from complete dataset (snapshot dt=%s)", analysis_dt)
    state_by_source = (
        complete_df.groupby(["source", "status"]).size().reset_index(name="count")
    )
    for _, row in state_by_source.sort_values(["source", "status"]).iterrows():
        logger.info(
            "   source=%s | status=%s | count=%s",
            row["source"],
            row["status"],
            f"{int(row['count']):,}",
        )

    state_global = complete_df["status"].value_counts(dropna=False)
    logger.info("KPI - global state counts")
    for status_name, count in state_global.items():
        logger.info("   status=%s | count=%s", status_name, f"{int(count):,}")

    if kpis_df is None or len(kpis_df) == 0:
        logger.warning("Generated KPI dataset is empty or unreadable")
        return

    if "snapshot_dt" in kpis_df.columns:
        latest_kpis = kpis_df[kpis_df["snapshot_dt"] == analysis_dt].copy()
        if len(latest_kpis) == 0:
            latest_kpis = kpis_df.sort_values("snapshot_dt").groupby("source").tail(1)
    else:
        latest_kpis = kpis_df.copy()

    logger.info("KPI - daily metrics by source (latest available)")
    for _, row in latest_kpis.sort_values("source").iterrows():
        logger.info(
            (
                "   source=%s | active=%s | published=%s | unpublished=%s | reappeared=%s | "
                "new=%s | unpublished_events=%s | reappeared_events=%s | churn_rate=%s | cum_reappearance_rate=%s"
            ),
            row.get("source", "unknown"),
            f"{int(row.get('active_stock_count', 0)):,}",
            f"{int(row.get('published_count', 0)):,}",
            f"{int(row.get('unpublished_count', 0)):,}",
            f"{int(row.get('reappeared_count', 0)):,}",
            f"{int(row.get('new_published_events_count', 0)):,}",
            f"{int(row.get('unpublished_events_count', 0)):,}",
            f"{int(row.get('reappeared_events_count', 0)):,}",
            f"{float(row.get('churn_rate')):.4f}" if pd.notna(row.get("churn_rate")) else "nan",
            (
                f"{float(row.get('cumulative_reappearance_rate')):.4f}"
                if pd.notna(row.get("cumulative_reappearance_rate"))
                else "nan"
            ),
        )


def _generate_offer_duration_chart(
    storage_analytics: Storage,
    complete_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    analysis_dt: str,
    output_prefix: Optional[str] = None,
) -> List[str]:
    """
    Generate charts for offer online duration (mean/median time from first_seen to last_seen).
    
    Duration is calculated from the complete historical timeline:
    - For each offer (id, source), finds the earliest and latest snapshot_dt exists
    - Calculates duration as: max(snapshot_dt) - min(snapshot_dt) in days
    - Only includes offers with duration > 0 (excludes offers first seen today)

    Charts produced:
    1. offer_duration_overview.png: global mean/median duration (offers with history only)
    2. offer_duration_by_rome.png: top 10 ROME codes by volume with mean/median durations
    """
    chart_keys: List[str] = []

    if complete_df is None or len(complete_df) == 0:
        logger.warning("Skipping duration chart generation: complete dataset is empty")
        return []
    
    if timeline_df is None or len(timeline_df) == 0:
        logger.warning("Skipping duration chart generation: timeline is empty")
        return []

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("Skipping duration chart generation: matplotlib unavailable (%s)", e)
        return []

    # Calculate historical daterange from timeline for each offer (id, source)
    logger.info("Recalculating duration from complete historical timeline...")
    timeline_copy = timeline_df.copy()
    timeline_copy["snapshot_dt"] = pd.to_datetime(timeline_copy["snapshot_dt"], errors="coerce")
    
    # Group by offer and find min/max snapshot_dt
    date_ranges = timeline_copy.groupby(["id", "source"])["snapshot_dt"].agg(["min", "max"]).reset_index()
    date_ranges.columns = ["id", "source", "first_seen_dt_actual", "last_seen_dt_actual"]
    
    # Calculate duration in days
    date_ranges["duration_days"] = (date_ranges["last_seen_dt_actual"] - date_ranges["first_seen_dt_actual"]).dt.total_seconds() / (24 * 3600)
    
    logger.info("Duration recalculated from timeline: %s offers have historical daterange", len(date_ranges))
    logger.info("  Duration stats: min=%s, max=%s, mean=%.2f, median=%.2f days",
                f"{date_ranges['duration_days'].min():.0f}",
                f"{date_ranges['duration_days'].max():.0f}",
                date_ranges['duration_days'].mean(),
                date_ranges['duration_days'].median())
    
    # Merge duration back to complete_df
    df = complete_df.copy()
    df = df.merge(date_ranges[["id", "source", "duration_days"]], on=["id", "source"], how="left")
    df["duration_days"] = df["duration_days"].fillna(0)

    # Parse date columns (for logging only, duration already calculated from timeline)
    if "rome_code" not in df.columns:
        logger.warning("Column 'rome_code' not found in dataset for ROME analysis")

    # Filter valid durations (> 0 - excludes new offers with no history)
    df = df[df["duration_days"] > 0].copy()

    if len(df) == 0:
        logger.warning("Skipping duration chart: no offers with historical data (all are new today)")
        return []
    
    logger.info("Duration analysis:")
    logger.info(f"  Offers with history: {len(df):,}")

    # 1) Global overview chart
    global_mean = df["duration_days"].mean()
    global_median = df["duration_days"].median()

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Histogram
    ax1.hist(df["duration_days"], bins=50, alpha=0.7, color="#3498db", edgecolor="black")
    ax1.axvline(global_mean, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {global_mean:.1f} days")
    ax1.axvline(global_median, color="#2ecc71", linestyle="--", linewidth=2, label=f"Median: {global_median:.1f} days")

    ax1.set_title("Offer Online Duration Distribution\n(Time from first seen to last seen)", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Duration (days)")
    ax1.set_ylabel("Number of Offers")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Stats text box
    stats_text = (
        f"Total Offers: {len(df):,}\n"
        f"Mean: {global_mean:.2f} days\n"
        f"Median: {global_median:.2f} days\n"
        f"Min: {df['duration_days'].min():.0f} days\n"
        f"Max: {df['duration_days'].max():.0f} days\n"
        f"Std: {df['duration_days'].std():.2f} days"
    )
    ax1.text(
        0.98, 0.97, stats_text, transform=ax1.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), family="monospace"
    )

    def _save_figure(fig, filename: str) -> str:
        key = f"dt={analysis_dt}/segment=plots/{filename}"
        if output_prefix:
            key = f"dt={analysis_dt}/segment=plots/{output_prefix}_{filename}"
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        storage_analytics.write_bytes(key, buffer.getvalue(), content_type="image/png")
        chart_keys.append(key)
        logger.info("Saved chart: %s", key)
        return key

    _save_figure(fig1, "offer_duration_overview.png")

    # 2) By top 10 ROME codes
    logger.info("Available columns in df: %s", sorted(df.columns.tolist()))
    
    if "rome_code" not in df.columns:
        logger.warning("Skipping ROME duration chart: rome_code column not found in complete dataset")
        logger.info("Looking for alternative columns containing 'rome'...")
        rome_cols = [c for c in df.columns if 'rome' in c.lower()]
        logger.info("Found columns with 'rome': %s", rome_cols)
        return chart_keys
    
    # Check for null values in rome_code
    rome_nulls = df["rome_code"].isna().sum()
    rome_non_nulls = df["rome_code"].notna().sum()
    logger.info("rome_code: non-null=%s, null=%s", rome_non_nulls, rome_nulls)
    
    if rome_non_nulls == 0:
        logger.warning("Skipping ROME duration chart: rome_code column is all NaN")
        return chart_keys

    # Load ROME referential to get job labels
    rome_ref_dict = {}
    try:
        from src.storage.storage import get_storage_from_env
        storage_bronze = get_storage_from_env("bronze", "france_travail")
        rome_key = "rome/rome_metiers.jsonl"
        rome_records = storage_bronze.read_jsonl(rome_key)
        rome_ref_dict = {rec["code"]: rec["libelle"] for rec in rome_records}
        logger.info("Loaded ROME referential: %s codes", len(rome_ref_dict))
    except Exception as e:
        logger.warning("Could not load ROME referential: %s (will use codes only)", e)

    # Top 10 ROME codes by volume (only in offers with history)
    rome_counts = df["rome_code"].value_counts()
    logger.info("Top 10 ROME codes (from offers with history): %s", rome_counts.head(10))
    
    top_10_romes = rome_counts.head(10).index.tolist()
    if len(top_10_romes) == 0:
        logger.warning("Skipping ROME duration chart: no valid ROME codes in offers with history")
        return chart_keys

    df_rome = df[df["rome_code"].isin(top_10_romes)].copy()
    logger.info("Filtered to top 10 ROME codes: %s rows (from %s offers with history)", len(df_rome), len(df))

    # Calculate mean and median by ROME
    rome_stats = df_rome.groupby("rome_code")["duration_days"].agg(["mean", "median", "count"]).reset_index()
    rome_stats = rome_stats.sort_values("count", ascending=False)
    
    # Add ROME labels
    rome_stats["rome_label"] = rome_stats["rome_code"].map(rome_ref_dict).fillna(rome_stats["rome_code"])
    # Truncate long labels for readability (max 40 chars)
    rome_stats["rome_display"] = rome_stats.apply(
        lambda r: f"{r['rome_code']}: {r['rome_label'][:37]}..." if len(r['rome_label']) > 40 else f"{r['rome_code']}: {r['rome_label']}", 
        axis=1
    )
    
    logger.info("ROME stats computed: %s rows\n%s", len(rome_stats), rome_stats[["rome_code", "rome_label", "mean", "median", "count"]])

    fig2, ax2 = plt.subplots(figsize=(14, 8))

    x_pos = np.arange(len(rome_stats))
    width = 0.35
    
    # Validate data before plotting
    mean_values = rome_stats["mean"].fillna(0).values
    median_values = rome_stats["median"].fillna(0).values
    logger.info("Mean values (filled NaN=0): min=%s, max=%s, any NaN=%s", 
                np.nanmin(mean_values), np.nanmax(mean_values), np.any(np.isnan(mean_values)))
    logger.info("Median values (filled NaN=0): min=%s, max=%s, any NaN=%s", 
                np.nanmin(median_values), np.nanmax(median_values), np.any(np.isnan(median_values)))

    bars1 = ax2.bar(x_pos - width/2, mean_values, width, label="Mean Duration", color="#e74c3c", alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, median_values, width, label="Median Duration", color="#2ecc71", alpha=0.8)

    # Add volume count labels on top (centered between bars)
    for i, (idx, row) in enumerate(rome_stats.iterrows()):
        max_bar_height = max(
            row["mean"] if pd.notna(row["mean"]) else 0,
            row["median"] if pd.notna(row["median"]) else 0
        )
        ax2.text(
            i, 
            max_bar_height + 0.15, 
            f"{int(row['count']):,} offers", 
            ha="center", 
            va="bottom", 
            fontsize=9,
            fontweight="bold"
        )

    ax2.set_title("Mean vs Median Offer Duration by Top 10 ROME Codes\n(Sorted by offer volume)", fontweight="bold", fontsize=13)
    ax2.set_xlabel("ROME Code & Job Category", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Duration (days)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(rome_stats["rome_display"], rotation=45, ha="right", fontsize=9)
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add some padding at the top for labels
    y_max = max(mean_values.max(), median_values.max()) if len(mean_values) > 0 else 5
    ax2.set_ylim(0, y_max * 1.15)
    
    plt.tight_layout()

    _save_figure(fig2, "offer_duration_by_rome.png")

    return chart_keys


def _generate_temporal_plots(
    storage_analytics: Storage,
    kpis_df: pd.DataFrame,
    analysis_dt: str,
    output_prefix: Optional[str] = None,
) -> List[str]:
    """
    Generate temporal monitoring charts from KPI dataframe and save them as PNG.

    Charts produced:
    1. active_stock_over_time.png
    2. offers_status_stacked_by_source.png (published/unpublished/reappeared by source + global)
    3. lifecycle_rates_over_time.png (churn / cumulative reappearance)
    """
    if kpis_df is None or len(kpis_df) == 0:
        logger.warning("Skipping chart generation: KPI dataframe is empty")
        return []

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("Skipping chart generation: matplotlib unavailable (%s)", e)
        return []

    df = kpis_df.copy()
    if "snapshot_dt" not in df.columns or "source" not in df.columns:
        logger.warning("Skipping chart generation: KPI dataframe missing required columns")
        return []

    df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"], errors="coerce")
    df = df.dropna(subset=["snapshot_dt"])
    if len(df) == 0:
        logger.warning("Skipping chart generation: no valid snapshot_dt values")
        return []

    chart_keys: List[str] = []

    def _save_figure(fig, filename: str) -> str:
        key = f"dt={analysis_dt}/segment=plots/{filename}"
        if output_prefix:
            key = f"dt={analysis_dt}/segment=plots/{output_prefix}_{filename}"
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        storage_analytics.write_bytes(key, buffer.getvalue(), content_type="image/png")
        chart_keys.append(key)
        logger.info("Saved chart: %s", key)
        return key

    # 1) Active stock trend by source
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for source, grp in df.sort_values("snapshot_dt").groupby("source"):
        ax1.plot(
            grp["snapshot_dt"],
            grp.get("active_stock_count", pd.Series([0] * len(grp), index=grp.index)),
            marker="o",
            linewidth=2,
            label=str(source),
        )
    ax1.set_title("Active Stock Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Active offers")
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="Source")
    _save_figure(fig1, "active_stock_over_time.png")

    # 1b) Stacked bar chart: published/unpublished/reappeared by source and date
    df_sorted = df.sort_values("snapshot_dt")
    pivot_published = df_sorted.pivot_table(
        index="snapshot_dt", columns="source", values="published_count", aggfunc="sum"
    )
    pivot_unpublished = df_sorted.pivot_table(
        index="snapshot_dt", columns="source", values="unpublished_count", aggfunc="sum"
    )
    pivot_reappeared = df_sorted.pivot_table(
        index="snapshot_dt", columns="source", values="reappeared_count", aggfunc="sum"
    )

    fig_stack, ((ax_s1, ax_s2), (ax_s_total, ax_legend)) = plt.subplots(
        2, 2, figsize=(14, 10)
    )
    ax_legend.axis("off")

    sources = pivot_published.columns.tolist()
    x_pos = np.arange(len(pivot_published.index))
    width = 0.35

    colors = {"published": "#2ecc71", "unpublished": "#e74c3c", "reappeared": "#3498db"}

    for idx, source in enumerate(sources):
        ax = ax_s1 if idx == 0 else ax_s2

        # Set value in a panda series with the same index as pivot tables, 
        # filling  with n [0] fo missng
        pub = pivot_published.get(source, pd.Series([0] * len(pivot_published)))
        unpub = pivot_unpublished.get(source, pd.Series([0] * len(pivot_unpublished)))
        reapp = pivot_reappeared.get(source, pd.Series([0] * len(pivot_reappeared)))

        x = x_pos + (idx * width)
        ax.bar(x, 
               pub, 
               width, 
               label="Published", 
               color=colors["published"], 
               alpha=0.8)
        
        ax.bar(
            x,
            unpub,
            width,
            bottom=pub,
            label="Unpublished",
            color=colors["unpublished"],
            alpha=0.8,
        )

        ax.bar(
            x,
            reapp,
            width,
            bottom=pub + unpub,
            label="Reappeared",
            color=colors["reappeared"],
            alpha=0.8,
        )

        ax.set_title(f"Offers Status - {source}", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Offers")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [dt.strftime("%Y-%m-%d") for dt in pivot_published.index], rotation=45, ha="right"
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper left", fontsize=8)

    # Global totals
    global_published = pivot_published.sum(axis=1)
    global_unpublished = pivot_unpublished.sum(axis=1)
    global_reappeared = pivot_reappeared.sum(axis=1)

    x_global = np.arange(len(global_published))

    ax_s_total.bar(x_global, 
                   global_published, 
                   label="Published", 
                   color=colors["published"], 
                   alpha=0.8)

    ax_s_total.bar(
        x_global,
        global_unpublished,
        bottom=global_published,
        label="Unpublished",
        color=colors["unpublished"],
        alpha=0.8,
    )

    ax_s_total.bar(
        x_global,
        global_reappeared,
        bottom=global_published + global_unpublished,
        label="Reappeared",
        color=colors["reappeared"],
        alpha=0.8,
    )

    ax_s_total.set_title("Global Offers Status (All Sources)", fontweight="bold", fontsize=12)
    ax_s_total.set_xlabel("Date")
    ax_s_total.set_ylabel("Number of Offers")
    ax_s_total.set_xticks(x_global)
    ax_s_total.set_xticklabels(
        [dt.strftime("%Y-%m-%d") for dt in pivot_published.index], rotation=45, ha="right"
    )
    ax_s_total.grid(True, alpha=0.3, axis="y")
    ax_s_total.legend(loc="upper left")

    # Legend on the 4th subplot
    legend_text = (
        "Chart Legend:\n\n"
        "Published (green): Active offers currently online\n"
        "Unpublished (red): Offers that were removed/inactive\n"
        "Reappeared (blue): Offers that came back after being inactive\n\n"
        "Total Active (green line): Published + Reappeared\n"
        "Total Inactive (red line): Unpublished\n"
    )
    ax_legend.text(0.1, 0.5, legend_text, fontsize=10, verticalalignment="center", family="monospace")

    plt.tight_layout()
    _save_figure(fig_stack, "offers_status_stacked_by_source.png")

    # 2) Lifecycle rates trend by source
    fig3, (ax31, ax32) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for source, grp in df.sort_values("snapshot_dt").groupby("source"):
        # Convert churn_rate to percentage
        churn_pct = grp["churn_rate"] * 100
        ax31.plot(grp["snapshot_dt"], churn_pct, marker="o", linewidth=2, label=str(source))
        
        # Convert cumulative_reappearance_rate to percentage
        reappear_pct = grp["cumulative_reappearance_rate"] * 100
        ax32.plot(grp["snapshot_dt"], reappear_pct, marker="o", linewidth=2, label=str(source))

    ax31.set_title("Churn Rate Over Time\n(% of active offers from previous day that became inactive)", fontsize=11, fontweight='bold')
    ax31.set_ylabel("Churn (%)")
    ax31.grid(True, alpha=0.3)
    ax31.legend(title="Source")
    ax31.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    ax32.set_title("Cumulative Reappearance Rate Over Time\n(% of offers that reappear after being unpublished)", fontsize=11, fontweight='bold')
    ax32.set_xlabel("Date")
    ax32.set_ylabel("Reappearance Rate (%)")
    ax32.grid(True, alpha=0.3)
    ax32.legend(title="Source")
    ax32.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    _save_figure(fig3, "lifecycle_rates_over_time.png")

    return chart_keys


def run_status_evolution_parquet_generation(
    storage_merged: Optional[Storage] = None,
    storage_status: Optional[Storage] = None,
    storage_analytics: Optional[Storage] = None,
    output_prefix: Optional[str] = None,
) -> Dict:
    """
    Phase 1: generate dataset parquets only (complete_dataset + status_timeline).

    This function generates the raw data parquets without computing KPIs or charts.
    All analytical computations (KPIs, visualizations) are deferred to Phase 2.
    """
    logger.info("=" * 80)
    logger.info("STATUS EVOLUTION ANALYTICS - PHASE 1: DATASET PARQUET GENERATION")
    logger.info("=" * 80)

    job_id = f"status-analytics-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    start_time = time.time()

    if storage_merged is None:
        storage_merged = get_storage_from_env("silver", "merged")
        logger.info("Storage (merged): silver/merged")

    if storage_status is None:
        storage_status = get_storage_from_env("silver", "status_history")
        logger.info("Storage (status_history): silver/status_history")

    if storage_analytics is None:
        storage_analytics = get_storage_from_env("silver", "status_analytics")
        logger.info("Storage (analytics): silver/status_analytics")

    try:
        log_to_db(
            endpoint="analyze_status_evolution",
            level="INFO",
            message=f"Start status evolution analytics Phase 1: {job_id}",
            task_id=job_id,
            status="RUNNING",
        )
    except Exception as e:
        logger.warning("[analyze_status_evolution Phase 1] log_to_db start failed: %s", e)

    # Load the complete status timeline to get the latest snapshot date
    # and build the complete latest-state dataset
    # Ex : 
    #  - latest_status_dt = "2026-03-10" (if latest snapshot in timeline is dt=2026-03-10)
    #  - latest_status_key = "dt=2026-03-10/segment=status_timeline/status_timeline.parquet"
    #  - timeline_df = full historical timeline with all snapshots with a "snapshot_dt" column
    timeline_df, latest_status_dt, latest_status_key = _load_status_timeline(storage_status)
    
    # Get the latest snapshot from the timeline to build the complete latest-state dataset
    latest_status_df = timeline_df[timeline_df["snapshot_dt"] == latest_status_dt].copy()

    # Extract the latest merged dataset (full state of all offers at the latest snapshot date)
    #  ex : 
    # - latest_merged_dt = "2026-03-10" (if latest snapshot in merged is dt=2026-03-10)
    # - latest_merged_key = "merged_dt=2026-03-10_ft_dt=2026-03-09_wttj_dt=2026-03-09.parquet"
    latest_merged_df, latest_merged_dt, latest_merged_key = _load_latest_merged(storage_merged)

    # complete_latest_df is the last dataset with the complete state of all offers 
    # at the latest snapshot date, enriched with status and timeline info.
    logger.info("Building complete latest-state dataset...")
    complete_latest_df = _build_complete_latest_dataset(
        latest_status_df=latest_status_df,
        latest_status_dt=latest_status_dt,
        latest_merged_df=latest_merged_df,
        latest_merged_dt=latest_merged_dt,
    )

    # Set Keys for saving outputs in analytics storage, partitioned by latest_status_dt
    analysis_dt = latest_status_dt
    # Complete dataset with latest state of all offers at snapshot date.
    complete_key = f"dt={analysis_dt}/segment=complete_dataset/complete_offers_with_status.parquet"
    # Timeline with full historical timeline of all offers (with snapshot_dt column for temporal analysis).
    timeline_key = f"dt={analysis_dt}/segment=status_timeline/status_timeline.parquet"

    # Add output prefix to keys if provided (allows to run multiple times a day without overwriting)
    if output_prefix:
        complete_key = f"dt={analysis_dt}/segment=complete_dataset/{output_prefix}_complete.parquet"
        timeline_key = f"dt={analysis_dt}/segment=status_timeline/{output_prefix}_timeline.parquet"

    # Save generated datasets as parquet in analytics storage
    logger.info("Saving dataset parquet outputs...")
    ok_complete = storage_tools.save_parquet_dataset(storage_analytics, complete_latest_df, complete_key)
    ok_timeline = storage_tools.save_parquet_dataset(storage_analytics, timeline_df, timeline_key)

    elapsed = time.time() - start_time
    success = ok_complete and ok_timeline

    # Generate a run.json metadata file.
    run_key = f"dt={analysis_dt}/run_id={job_id}/run.json"
    run_payload = {
        "job_id": job_id,
        "analysis_dt": analysis_dt,
        "timestamp_start": datetime.fromtimestamp(start_time).isoformat(),
        "timestamp_end": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "phase": "parquet_generation",
        "inputs": {
            "latest_status_key": latest_status_key,
            "latest_merged_key": latest_merged_key,
        },
        "outputs": {
            "complete_key": complete_key,
            "timeline_key": timeline_key,
        },
        "rows": {
            "complete_latest": int(len(complete_latest_df)),
            "timeline": int(len(timeline_df)),
        },
        "success": success,
    }

    try:
        storage_analytics.write_json(run_key, run_payload)
    except Exception as e:
        logger.warning("Failed to write analytics run.json: %s", e)

    if success:
        logger.info("Phase 1: Dataset parquet generation completed successfully")
        logger.info("Complete dataset rows: %s", f"{len(complete_latest_df):,}")
        logger.info("Timeline rows: %s", f"{len(timeline_df):,}")

        try:
            log_to_db(
                endpoint="analyze_status_evolution",
                level="INFO",
                message=(
                    f"Status analytics Phase 1 completed: complete={len(complete_latest_df):,}, "
                    f"timeline={len(timeline_df):,}"
                ),
                task_id=job_id,
                status="SUCCESS",
                duration_sec=elapsed,
                records_count=int(len(timeline_df)),
                output_key=complete_key,
            )
        except Exception as e:
            logger.warning("[analyze_status_evolution Phase 1] log_to_db success failed: %s", e)
    else:
        logger.error("Phase 1: Analytics failed while saving dataset outputs")
        try:
            log_to_db(
                endpoint="analyze_status_evolution",
                level="ERROR",
                message="Phase 1: Failed to save dataset outputs",
                task_id=job_id,
                status="ERROR",
                duration_sec=elapsed,
            )
        except Exception as e:
            logger.warning("[analyze_status_evolution Phase 1] log_to_db error failed: %s", e)

    return {
        "success": success,
        "job_id": job_id,
        "analysis_dt": analysis_dt,
        "complete_key": complete_key,
        "timeline_key": timeline_key,
        "phase": "parquet_generation",
        "run_key": run_key,
        "rows_complete": int(len(complete_latest_df)),
        "rows_timeline": int(len(timeline_df)),
        "elapsed_sec": elapsed,
    }


def run_status_evolution_analytics_computation(
    storage_analytics: Optional[Storage] = None,
    output_prefix: Optional[str] = None,
) -> Dict:
    """
    Phase 2: compute KPIs and generate temporal charts from latest timeline parquet.

    This phase auto-discovers the latest status_timeline parquet, computes daily KPIs,
    and generates temporal monitoring charts. It is completely independent of Phase 1
    and can be run separately at any time.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("STATUS EVOLUTION ANALYTICS - PHASE 2: COMPUTATION & VISUALIZATION")
    logger.info("=" * 80)

    if storage_analytics is None:
        storage_analytics = get_storage_from_env("silver", "status_analytics")
        logger.info("Storage (analytics): silver/status_analytics")

    # Auto-discover latest dt from status_analytics 
    # in storage_analytics keys
    all_keys = [k for k in storage_analytics.list_keys("") if k.endswith(".parquet")]
    if not all_keys:
        logger.warning("Phase 2: No parquet files found in silver/status_analytics")
        return {
            "success": False,
            "phase": "analytics_computation",
            "message": "No parquet files found in silver/status_analytics",
        }
    
    # Get last one dt=YYYY-MM-DD 
    all_dts = sorted({dt for dt in (_extract_dt_from_key(k) for k in all_keys) if dt})
    if not all_dts:
        logger.warning("Phase 2: No dt=YYYY-MM-DD found in analytics keys")
        return {
            "success": False,
            "phase": "analytics_computation",
            "message": "No dt partition found in silver/status_analytics",
        }
    
    # Latest dt is the one we will analyze in this run
    analysis_dt = all_dts[-1]
    logger.info("Auto-discovered latest analysis_dt: %s", analysis_dt)

    # Get all keys realted to the analyzed dt and find the timeline parquet 
    dt_keys = [k for k in all_keys if f"dt={analysis_dt}/" in k]

    # Timeline parquet ( required for KPI computation)
    timeline_candidates = [k for k in dt_keys if "/segment=status_timeline/" in k]

    # If multiple timeline candidates, 
    # try to find one with the output prefix (allows to run multiple times a day without overwriting)
    if output_prefix:
        preferred_timeline = [k for k in timeline_candidates if f"/{output_prefix}_" in k]
        if preferred_timeline:
            timeline_candidates = preferred_timeline

    if not timeline_candidates:
        logger.warning(
            "Phase 2: No timeline parquet found for latest dt=%s with prefix=%s",
            analysis_dt,
            output_prefix or "none",
        )
        return {
            "success": False,
            "phase": "analytics_computation",
            "message": f"No timeline parquet found for dt={analysis_dt}",
            "analysis_dt": analysis_dt,
        }
    # Timeline key foud, ex : 'dt=2026-03-10/segment=status_timeline/status_timeline.parquet' 
    timeline_key = sorted(timeline_candidates)[-1]
    logger.info("Loading timeline parquet: %s", timeline_key)

    # Load timeline parquet dataset (full historical timeline with snapshot_dt column for temporal analysis)
    timeline_df = storage_tools.load_parquet_dataset(storage_analytics, timeline_key)
    if timeline_df is None or len(timeline_df) == 0:
        logger.warning("Phase 2: Failed to load timeline parquet")
        return {
            "success": False,
            "phase": "analytics_computation",
            "message": f"Failed to load timeline parquet: {timeline_key}",
            "analysis_dt": analysis_dt,
        }

    start_time = time.time()
    logger.info("Computing daily status KPIs from timeline...")
    kpis_df = _compute_daily_status_kpis(timeline_df)
    
    # Save KPIs parquet (partitioned by dt=analysis_dt)
    kpis_key = f"dt={analysis_dt}/segment=kpis/daily_status_kpis.parquet"
    if output_prefix:
        kpis_key = f"dt={analysis_dt}/segment=kpis/{output_prefix}_kpis.parquet"

    logger.info("Saving KPI parquet...")
    ok_kpis = storage_tools.save_parquet_dataset(storage_analytics, kpis_df, kpis_key)

    if not ok_kpis:
        logger.error("Phase 2: Failed to save KPI parquet")
        return {
            "success": False,
            "phase": "analytics_computation",
            "message": "Failed to save KPI parquet",
            "analysis_dt": analysis_dt,
        }

    # Try to load complete dataset for logging if available
    complete_candidates = [k for k in dt_keys if "/segment=complete_dataset/" in k]
    if output_prefix:
        preferred_complete = [k for k in complete_candidates if f"/{output_prefix}_" in k]
        if preferred_complete:
            complete_candidates = preferred_complete

    complete_key = sorted(complete_candidates)[-1] if complete_candidates else None

    logger.info("KPI parquet rows: %s", f"{len(kpis_df):,}")
    _log_generated_outputs_kpis(
        storage_analytics=storage_analytics,
        complete_key=complete_key,
        kpis_key=kpis_key,
        analysis_dt=analysis_dt,
    )

    chart_keys: List[str] = []
    try:
        logger.info("Generating temporal monitoring charts...")
        chart_keys = _generate_temporal_plots(
            storage_analytics=storage_analytics,
            kpis_df=kpis_df,
            analysis_dt=analysis_dt,
            output_prefix=output_prefix,
        )
        logger.info("Phase 2: Generated %s chart files from KPI data", len(chart_keys))
    except Exception as e:
        logger.warning("Phase 2: KPI chart generation failed (non-blocking): %s", e)

    # Generate offer duration charts
    try:
        logger.info("Generating offer duration charts...")
        if complete_key:
            complete_df = storage_tools.load_parquet_dataset(storage_analytics, complete_key)
            if complete_df is not None and len(complete_df) > 0:
                duration_chart_keys = _generate_offer_duration_chart(
                    storage_analytics=storage_analytics,
                    complete_df=complete_df,
                    timeline_df=timeline_df,
                    analysis_dt=analysis_dt,
                    output_prefix=output_prefix,
                )
                chart_keys.extend(duration_chart_keys)
                logger.info("Phase 2: Generated %s chart files from offer duration data", len(duration_chart_keys))
            else:
                logger.warning("Phase 2: Complete dataset unavailable for duration charts")
        else:
            logger.warning("Phase 2: Complete dataset key not found for duration charts")
    except Exception as e:
        logger.warning("Phase 2: Duration chart generation failed (non-blocking): %s", e)

    elapsed = time.time() - start_time

    return {
        "success": True,
        "phase": "analytics_computation",
        "analysis_dt": analysis_dt,
        "timeline_key": timeline_key,
        "kpis_key": kpis_key,
        "chart_keys": chart_keys,
        "rows_kpis": int(len(kpis_df)),
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build status evolution analytics: Phase 1 generates dataset parquets, "
        "Phase 2 computes KPIs and visualization"
    )
    # Compatibility with existing debug launch profiles that still pass status mode args.
    # They are ignored here because this script has a single analytics execution mode.
    parser.add_argument(
        "--mode",
        default=None,
        help="Ignored in this script (kept for compatibility with shared launch configs)",
    )
    parser.add_argument(
        "--compute-status",
        default=None,
        help="Ignored in this script (kept for compatibility with calculate_offer_status args)",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output filename prefix for analytics parquet files",
    )
    parser.add_argument(
        "--skip-computation",
        action="store_true",
        default=False,
        help="Only generate dataset parquets (Phase 1), skip KPI computation and charts (Phase 2)",
    ),
    parser.add_argument(
        "--skip-analytics-computation",
        action="store_true",
        default=False,
        help="Skip KPI computation and chart generation (Phase 2) - can be used to only run Phase 1",
    )

    args, unknown_args = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if unknown_args:
        logger.warning("Ignoring unknown CLI args: %s", " ".join(unknown_args))
    if args.mode or args.compute_status:
        logger.info(
            "Compatibility args received and ignored: mode=%s compute-status=%s",
            args.mode,
            args.compute_status,
        )

    # Phase 1: Dataset parquet generation (always runs)
    if not args.skip_computation:
        try:
            # Phase 1 is critical for generating the necessary parquets. 
            # It generates the complete dataset and timeline parquets that Phase 2 depends on :
            # - complete dataset parquet (with offer details and status)
            # - timeline parquet (with daily status snapshots)
            parquet_result = run_status_evolution_parquet_generation(
                output_prefix=args.output_prefix,
            )
        except Exception as e:
            logger.error("Phase 1 failed with exception: %s", e)
            parquet_result = {"success": False, "error": str(e)}

        if not parquet_result.get("success"):
            logger.error("Phase 1 failed: %s", parquet_result.get("error", "unknown error"))
            logger.warning("Skipping Phase 2 due to Phase 1 failure.")

    # Phase 2: KPI computation and visualization (optional, independent)
    if not args.skip_analytics_computation:
        logger.info("")
        logger.info("Running Phase 2 (KPI computation and visualization)...")
        computation_result = run_status_evolution_analytics_computation(
            output_prefix=args.output_prefix,
        )
        if not computation_result.get("success"):
            logger.warning("Phase 2 failed (non-blocking).")
    else:
        logger.info("Phase 2 skipped (--skip-analytics-computation)")

    raise SystemExit(0)


if __name__ == "__main__":
    main()
