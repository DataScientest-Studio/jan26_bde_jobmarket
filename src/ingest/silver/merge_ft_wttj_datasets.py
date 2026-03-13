"""
Merge canonical FT and WTTJ silver datasets into a unified dataset.

Responsibilities of this file:
- Read already-normalized silver datasets for FT and WTTJ.
- Apply merge-only logic: concatenation, deduplication, and cross-source harmonization.
- Persist the merged output dataset and publish summary statistics.

Out of scope for this file:
- Source-specific parsing/cleaning from bronze payloads.
- Source-specific enrichment (for example WTTJ ROME inference).

Usage:
    python -m src.ingest.silver.merge_ft_wttj_datasets
    python -m src.ingest.silver.merge_ft_wttj_datasets --format parquet
    python -m src.ingest.silver.merge_ft_wttj_datasets --output-prefix datasets/custom
"""
import argparse
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import orjson  # 10x faster than standard json
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False
    
import pandas as pd
from dotenv import load_dotenv

from src.storage.storage import get_storage_from_env
from src.ingest.data_models.silver_datamodel_class import Silver_Datamodel
from src.utils.normalization_helpers import write_dataframe_with_format
from src.utils import find_latest_data_prefix, normalize_list_to_strings
import src.utils.merge_dataset_utils as merge_dataset_utils

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================
# Configuration
# =============================
FT_SILVER_PREFIX = os.getenv("FT_SILVER_PREFIX", "dt=2026-02-18/segment=jobs")
WTTJ_SILVER_PREFIX = os.getenv("WTTJ_SILVER_PREFIX", "dt=2026-02-22/segment=jobs")
MERGED_DATASET_PREFIX  = os.getenv("MERGED_DATASET_PREFIX", "datasets/ft_wttj_merged")

_FORMAT_FROM_EXT = {"parquet": "parquet", "jsonl": "jsonl", "csv": "csv"}

# =============================
# File Processing Helpers
# =============================
def _read_source_file(storage, key: str, mode: str) -> Any:
    """
    Generic helper to process a file depending on mode.

    Modes:
    - parquet_df: returns pd.DataFrame
    """
    if mode == "parquet_df":
        try:
            return storage.read_parquet(key)
        except Exception as e:
            logger.error(f"   ❌ Erreur lecture {key}: {e}")
            return pd.DataFrame()

    raise ValueError(f"Unsupported processing mode: {mode}")

# =============================
# Reading normalized silver inputs
# =============================
def read_silver_parquet_data(storage, prefix: str, source_label: str) -> pd.DataFrame:
    """
    Read normalized source data from silver parquet files with parallelization.
    """
    logger.info(f"📂 Reading {source_label} Silver: {prefix}")
    
    keys = list(storage.list_keys(prefix))
    parquet_keys = [k for k in keys if k.endswith('.parquet')]
    logger.info(f"   Found {len(parquet_keys)} Parquet files")
    
    if len(parquet_keys) == 0:
        logger.warning(f"⚠️ No Parquet files found in {prefix}")
        return pd.DataFrame()
    
    max_workers = min(int(os.getenv("SILVER_READ_WORKERS", "20")), len(parquet_keys))
    dfs = []

    if len(parquet_keys) > 1:
        logger.info(f"   Using {max_workers} workers for parallelization")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_read_source_file, storage, key, mode="parquet_df"): key for key in parquet_keys}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    df = future.result()
                    if not df.empty:
                        dfs.append(df)
                    logger.info(f"   Progress: {i}/{len(parquet_keys)} files")
                except Exception as e:
                    key = futures[future]
                    logger.error(f"   ❌ Error processing {key}: {e}")
    else:
        try:
            df = _read_source_file(storage, parquet_keys[0], mode="parquet_df")
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.error(f"   ❌ Error reading {parquet_keys[0]}: {e}")

    if not dfs:
        logger.warning(f"⚠️ No {source_label} data found")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"✅ {source_label}: {len(df):,} offers loaded")
    return df


def harmonize_silver_schema(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Ensure canonical merge columns exist and preserve existing normalized fields."""
    canonical_columns = Silver_Datamodel.get_dataframe_columns()
    normalized_columns = [
        "contract_normalized",
        "contract_detail",
        "experience_normalized",
        "experience_index",
        "experience_detail",
    ]
    if len(df) == 0:
        return pd.DataFrame(columns=canonical_columns + normalized_columns)

    default_values = {
        "id": "",
        "source": source_label,
        "url": "",
        "title": "",
        "description": "",
        "published_at": None,
        "updated_at": None,
        "unpublished_at": None,
        "status": "",
        "rome_code": "",
        "rome_label": "",
        "title_description": "",
        "contract_type": "",
        "worktime": "",
        "experience_level": "",
        "experience_description": "",
        "naf_code": "",
        "job_city": "",
        "job_postal_code": "",
        "company_name": "",
        "company_city": "",
        "company_postal_code": "",
        "company_url": "",
        "salary_min": 0.0,
        "salary_max": 0.0,
        "periodicity": None,
        "yearly_min": None,
        "yearly_max": None,
        "profile": "",
        "location_department": "",
        "skills": [],
        "salary_periodicity": "",
    }

    for col in canonical_columns:
        if col not in df.columns:
            df[col] = default_values[col]

    df["source"] = source_label
    df["skills"] = df["skills"].apply(lambda x: normalize_list_to_strings(x) if x is not None else [])

    # Keep normalized columns if already available in source datasets.
    if "contract_normalized" not in df.columns:
        df["contract_normalized"] = df["contract_type"]
    if "contract_detail" not in df.columns:
        df["contract_detail"] = None
    if "experience_normalized" not in df.columns:
        df["experience_normalized"] = df["experience_level"]
    if "experience_index" not in df.columns:
        df["experience_index"] = None
    if "experience_detail" not in df.columns:
        df["experience_detail"] = df["experience_description"]

    return df[canonical_columns + normalized_columns]

# =============================
# Merge and deduplication
# =============================
def merge_and_deduplicate(df_ft: pd.DataFrame, df_wttj: pd.DataFrame) -> pd.DataFrame:
    """
    Merge FT and WTTJ and deduplicate by URL.
    
    Args:
        df_ft: DataFrame France Travail
        df_wttj: DataFrame Welcome To The Jungle
        
    Returns:
        Merged and deduplicated DataFrame
    """
    logger.info("🔄 Merging datasets...")
    
    # Check if DataFrames are empty
    if len(df_ft) == 0 and len(df_wttj) == 0:
        logger.warning("⚠️ No data to merge, both DataFrames are empty")
        return pd.DataFrame()
    
    if len(df_ft) == 0:
        logger.warning("⚠️ No FT data, using only WTTJ")
        return df_wttj
    
    if len(df_wttj) == 0:
        logger.warning("⚠️ No WTTJ data, using only FT")
        return df_ft
    
    # Debug: check columns
    logger.info(f"   FT columns: {list(df_ft.columns)}")
    logger.info(f"   WTTJ columns: {list(df_wttj.columns)}")
    
    # Concatenate before deduplication to maximize chances of catching duplicates
    df_merged = pd.concat([df_ft, df_wttj], ignore_index=True)
    logger.info(f"   Total before deduplication: {len(df_merged):,} offers")
    
    # DDeduplication by URL (priority to FT if duplicates found)
    initial_count = len(df_merged)
    
    # Check if 'url' column exists before deduplication
    if 'url' in df_merged.columns:
        # Remove empty URLs
        df_merged = df_merged[df_merged['url'].notna() & (df_merged['url'] != "")]
        # Deduplicate by URL
        df_merged = df_merged.drop_duplicates(subset=['url'], keep='first')
    else:
        logger.warning("⚠️ 'url' column not found, deduplication impossible")
    
    duplicates_removed = initial_count - len(df_merged)
    logger.info(f"   Duplicates or empty URLs removed: {duplicates_removed:,}")
    logger.info(f"   Total after empty or duplicate removal: {len(df_merged):,} offers")
    
    # Filter offers without ROME code
    df_with_rome = df_merged[df_merged['rome_code'].notna() & (df_merged['rome_code'] != "")]
    df_without_rome = df_merged[df_merged['rome_code'].isna() | (df_merged['rome_code'] == "")]
    
    logger.info(f"   With ROME code: {len(df_with_rome):,} offers ({len(df_with_rome)/len(df_merged)*100:.1f}%)")
    logger.info(f"   Without ROME code: {len(df_without_rome):,} offers ({len(df_without_rome)/len(df_merged)*100:.1f}%)")
    
    return df_merged

# =============================
# Save dataset
# =============================
def save_pd_to_storage_with_format(df: pd.DataFrame, storage, format: str = "parquet", dt_ft: Optional[str] = None, dt_wttj: Optional[str] = None) -> str:
    """
    Save the merged dataset.
    
    Args:
        df: DataFrame to save
        storage: Storage instance
        format: Output format (parquet, jsonl, csv)
        dt_ft: Date from FT Silver prefix (for filename)
        dt_wttj: Date from WTTJ Silver prefix (for filename)
        
    Returns:
        Output key where data was saved
    """
    # Use the most recent date between FT and WTTJ instead of current date,
    # so re-running a historical import produces the same deterministic key.
    available_dates = [d for d in [dt_ft, dt_wttj] if d]
    dt_merged = max(available_dates) if available_dates else datetime.now().strftime("%Y-%m-%d")

    key_base = f"merged_dt={dt_merged}_ft_dt={dt_ft}_wttj_dt={dt_wttj}"
    output_key = write_dataframe_with_format(storage, df, key_base, format)
    logger.info(f"💾 Save {format.upper()}: {output_key}")
    
    logger.info(f"✅ Dataset saved: {output_key}")
    logger.info(f"   Size: {len(df):,} offers")
    
    return output_key


# =============================
# Main
# =============================
def merge_ft_wttj_datasets(
    ft_prefix: Optional[str] = None,
    wttj_prefix: Optional[str] = None,
    output_prefix: Optional[str] = None,
    output_format: str = "parquet",
    progress_callback: Optional[Any] = None,
) -> Dict:
    """
    Main function to merge datasets.
    
    Args:
        ft_prefix: FT Silver data prefix
        wttj_prefix: WTTJ Silver data prefix
        output_prefix: Output prefix
        output_format: Output format
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dict with success status, stats, and output key
    """
    start_time = time.perf_counter()
    # Storages
    storage_ft = get_storage_from_env("silver", "france_travail")
    storage_wttj = get_storage_from_env("silver", "welcometothejungle")
    
    # Auto-detection of prefixes if not specified
    if not ft_prefix:
        ft_prefix = os.getenv("FT_SILVER_PREFIX")
        if not ft_prefix:
            logger.info("🔍 Auto-detection of FT Silver prefix...")
            if progress_callback:
                progress_callback("detecting_ft_prefix", "Détection du préfixe FT...")
            ft_prefix = find_latest_data_prefix(storage_ft, "", "")
            if ft_prefix:
                logger.info(f"   ✓ Found: {ft_prefix}")
            else:
                logger.warning("   ⚠️ No FT Silver data found")
                ft_prefix = FT_SILVER_PREFIX
    
    if not wttj_prefix:
        wttj_prefix = os.getenv("WTTJ_SILVER_PREFIX")
        if not wttj_prefix:
            logger.info("🔍 Auto-detection of WTTJ Silver prefix...")
            if progress_callback:
                progress_callback("detecting_wttj_prefix", "Détection du préfixe WTTJ...")
            wttj_prefix = find_latest_data_prefix(storage_wttj, "", "")
            if wttj_prefix:
                wttj_prefix += "/segment=jobs"
                logger.info(f"   ✓ Found: {wttj_prefix}")
            else:
                logger.warning("   ⚠️ No WTTJ Silver data found")
                wttj_prefix = WTTJ_SILVER_PREFIX  # Fallback
    
    output_prefix = output_prefix or os.getenv("MERGED_DATASET_PREFIX", MERGED_DATASET_PREFIX)
    
    logger.info("=" * 80)
    logger.info("🚀 MERGING FT + WTTJ DATASETS WITH ROME CODES")
    logger.info("=" * 80)
    logger.info(f"📂 Source FT: {ft_prefix}")
    logger.info(f"📂 Source WTTJ: {wttj_prefix}")
    logger.info(f"📂 Destination: {output_prefix}")
    logger.info("")

    # Read source-normalized silver data for WTTJ
    if progress_callback:
        progress_callback("reading_wttj", "Lecture des données WTTJ...")
    df_wttj = read_silver_parquet_data(storage_wttj, wttj_prefix, "Welcome To The Jungle")
    
    if progress_callback:
        progress_callback("harmonizing_wttj", f"Harmonisation schéma WTTJ ({len(df_wttj):,} offres)...")
    df_wttj = harmonize_silver_schema(df_wttj, "WTTJ")

    # Read source-normalized silver data for FT
    if progress_callback:
        progress_callback("reading_ft", "Lecture des données France Travail...")
    df_ft = read_silver_parquet_data(storage_ft, ft_prefix, "France Travail")
    
    if progress_callback:
        progress_callback("harmonizing_ft", f"Harmonisation schéma FT ({len(df_ft):,} offres)...")
    df_ft = harmonize_silver_schema(df_ft, "FT")

    # Merge and deduplicate datasets
    if progress_callback:
        progress_callback("merging", "Fusion et déduplication...")
    df_merged = merge_and_deduplicate(df_ft, df_wttj)

    # =======================================
    # Apply normalization to mergeded dataset
    # =======================================
    df_merged = merge_dataset_utils.normalize_contracts(df_merged, merge_dataset_utils.PATTERNS_CONTRACT_NORMALIZE)
    logger.info("✅ Normalize Contracts COMPLETED ")

    # Create composite experience column for normalization
    # TODO : change column name in source
    df_merged['experience_source_composite'] = df_merged.apply(merge_dataset_utils.get_experience_col, axis=1)
    # TODO: Check normalisation - vérifier que experience_normalized produit bien les labels
    # attendus (Débutant, 0-1 an, 1-2 ans…) pour FT (codes E/D/S) et WTTJ (LESS_THAN_6_MONTHS…)
    # lors de la phase d'ingestion. Les parquets Silver existants peuvent contenir les codes bruts.
    # Normalize experience levels using the composite column
    df_merged = merge_dataset_utils.normalize_experience(df_merged,'experience_source_composite')
    logger.info("✅ Normalize Experience COMPLETED ")

    df_merged = merge_dataset_utils.normalize_salary(df_merged)
    logger.info("✅ Normalize Salary COMPLETED ")
    logger.info("✅ Normalize Yearly Salary COMPLETED ")
    # =======================================

    # Extract dt from prefixes for filename
    dt_ft_match = re.search(r'dt=([0-9\-]+)', ft_prefix)
    dt_wttj_match = re.search(r'dt=([0-9\-]+)', wttj_prefix)
    dt_ft = dt_ft_match.group(1) if dt_ft_match else None
    dt_wttj = dt_wttj_match.group(1) if dt_wttj_match else None
    
    # Save merged dataset 
    if progress_callback:
        progress_callback("saving", f"Sauvegarde du dataset ({len(df_merged):,} offres)...")
    storage_output = get_storage_from_env("silver", "merged")

    output_key = save_pd_to_storage_with_format(df_merged, storage_output, output_format, dt_ft, dt_wttj)

    # Statistics
    if progress_callback:
        progress_callback("computing_stats", "Calcul des statistiques...")
    try:
        merge_dataset_utils.print_statistics(df_merged)
    except UnicodeEncodeError as e:
        logger.warning(f"Statistics display skipped due to console encoding issue: {e}")
    
    end_time = time.perf_counter()
    elapsed_s = end_time - start_time
    
    logger.info("=" * 80)
    logger.info("✅ MERGING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    # Compute statistics for response
    rome_stats = df_merged[df_merged['rome_code'].notna() & (df_merged['rome_code'] != '')]
    source_counts = df_merged['source'].value_counts().to_dict()
    
    return {
        "success": True,
        "message": f"Fusion réussie: {len(df_merged):,} offres fusionnées",
        "output_key": output_key,
        "output_format": output_format,
        "ft_prefix": ft_prefix,
        "wttj_prefix": wttj_prefix,
        "total_offers": len(df_merged),
        "ft_offers": source_counts.get('FT', 0),
        "wttj_offers": source_counts.get('WTTJ', 0),
        "offers_with_rome": len(rome_stats),
        "unique_rome_codes": df_merged['rome_code'].nunique(),
        "elapsed_s": elapsed_s
    }

def replay_merge_from_key(merged_key: str, output_format: Optional[str] = None) -> Dict:
    """Replay a merge run from a merged dataset key or filename.

    Parses ft_dt, wttj_dt and the file extension from the key, then calls
    merge_ft_wttj_datasets with the corresponding Silver prefixes,
    producing the same deterministic output key as the original run.

    Args:
        merged_key: Key or filename of the form
            ``merged_dt=YYYY-MM-DD_ft_dt=YYYY-MM-DD_wttj_dt=YYYY-MM-DD.{parquet|jsonl|csv}``
        output_format: Output format override. If None, inferred from the key
            extension (.parquet → "parquet", .jsonl → "jsonl", .csv → "csv").
            Defaults to "parquet" if the extension is absent or unrecognised.

    Returns:
        Result dict from merge_ft_wttj_datasets.

    Raises:
        ValueError: If ft_dt or wttj_dt cannot be extracted from merged_key.

    Example:
        >>> replay_merge_from_key(
        ...     "merged_dt=2026-03-05_ft_dt=2026-03-03_wttj_dt=2026-03-05.parquet"
        ... )
    """
    ft_match = re.search(r"ft_dt=(\d{4}-\d{2}-\d{2})", merged_key)
    wttj_match = re.search(r"wttj_dt=(\d{4}-\d{2}-\d{2})", merged_key)

    if not ft_match or not wttj_match:
        raise ValueError(
            f"Cannot extract ft_dt / wttj_dt from key: {merged_key!r}. "
            "Expected format: merged_dt=..._ft_dt=YYYY-MM-DD_wttj_dt=YYYY-MM-DD"
        )

    ft_dt = ft_match.group(1)
    wttj_dt = wttj_match.group(1)

    if output_format is None:
        ext_match = re.search(r"\.(\w+)$", merged_key)
        ext = ext_match.group(1).lower() if ext_match else ""
        output_format = _FORMAT_FROM_EXT.get(ext, "parquet")

    ft_prefix = f"dt={ft_dt}"
    wttj_prefix = f"dt={wttj_dt}/segment=jobs"

    logger.info(
        "Replaying merge from key %r — ft_prefix=%s  wttj_prefix=%s  format=%s",
        merged_key, ft_prefix, wttj_prefix, output_format,
    )
    return merge_ft_wttj_datasets(
        ft_prefix=ft_prefix,
        wttj_prefix=wttj_prefix,
        output_format=output_format,
    )


def main():

    ret= replay_merge_from_key("merged_dt=2026-03-05_ft_dt=2026-03-03_wttj_dt=2026-03-05.parquet")
    ret= replay_merge_from_key("merged_dt=2026-03-07_ft_dt=2026-03-07_wttj_dt=2026-03-07.parquet")
    ret= replay_merge_from_key("merged_dt=2026-03-10_ft_dt=2026-03-09_wttj_dt=2026-03-09.parquet")

    exit(0)

    """Entry point for CLI"""
    parser = argparse.ArgumentParser(description="Merge FT + WTTJ datasets with ROME codes")
    parser.add_argument("--ft-prefix", help="FT Silver data prefix")
    parser.add_argument("--wttj-prefix", help="WTTJ Silver data prefix")
    parser.add_argument("--output-prefix", help="Output prefix")
    parser.add_argument("--format", choices=["parquet", "jsonl", "csv"], default="parquet", help="Output format")
    
    args = parser.parse_args()

    #wttj_prefix=args.wttj_prefix
    #ft_prefix=args.ft_prefix
    #output_prefix=args.output_prefix
    #output_format=args.format

    wttj_prefix= "dt=2026-03-09/segment=jobs"
    ft_prefix="dt=2026-03-09"
    output_prefix="merged"
    output_format="parquet"

    merge_ft_wttj_datasets(
        ft_prefix=ft_prefix,
        wttj_prefix=wttj_prefix,
        output_prefix=output_prefix,
        output_format=output_format,
    )


if __name__ == "__main__":
    main()
