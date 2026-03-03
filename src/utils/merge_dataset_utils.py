from src.config.env import load_project_env
load_project_env()  # safe à rappeler (idempotent)

from src.storage.storage import get_storage_from_env

import logging
logger = logging.getLogger(__name__)

import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_wttj_file(storage, key: str) -> pd.DataFrame:
    """
    Reads a single WTTJ Parquet file and returns a normalized DataFrame.
    Helper function for parallel processing.
    """
    try:
        return storage.read_parquet(key)
    except Exception as e:
        logger.error(f"   ❌ Erreur lecture {key}: {e}")
        return pd.DataFrame()

def read_wttj_parquet_file_to_df(storage, prefix: str) -> pd.DataFrame:
    """
    Reads Welcome To The Jungle data from the Silver layer (Parquet) with parallelization.
    
    Args:
        storage: Storage instance
        prefix: S3/local prefix for Silver WTTJ data
        
    Returns:
        DataFrame with normalized columns
    """
    logger.info(f"📂 Reading Welcome To The Jungle Silver: {prefix}")
    
    keys = list(storage.list_keys(prefix))
    parquet_keys = [k for k in keys if k.endswith('.parquet')]
    logger.info(f"   Found {len(parquet_keys)} Parquet files")
    
    if len(parquet_keys) == 0:
        logger.warning(f"⚠️ No Parquet files found in {prefix}")
        return pd.DataFrame()
    
    # Parallelization for Parquet files
    # With a pool of 50 connections, we can go up to 40 workers
    max_workers = min(int(os.getenv("WTTJ_READ_WORKERS", "40")), len(parquet_keys))
    dfs = []
    
    if len(parquet_keys) > 1:
        logger.info(f"   Using {max_workers} workers for parallelization")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_wttj_file, storage, key): key for key in parquet_keys}
            
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    df = future.result()
                    if not df.empty:
                        dfs.append(df)
                    logger.info(f"   WTTJ Progress: {i}/{len(parquet_keys)} files")
                except Exception as e:
                    key = futures[future]
                    logger.error(f"   ❌ Error processing {key}: {e}")
    else:
        # If only one file, no need for parallelization
        try:
            df = storage.read_parquet(parquet_keys[0])
            dfs.append(df)
        except Exception as e:
            logger.error(f"   ❌ Error reading {parquet_keys[0]}: {e}")
    
    if not dfs:
        logger.warning("⚠️ No WTTJ data found")
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"✅ Welcome To The Jungle: {len(df):,} offers loaded")
    return df    

def print_statistics(df: pd.DataFrame) -> None:
    """Print detailed statistics of the dataset"""
    
    print("\n" + "=" * 80)
    print("📊 STATISTICS OF THE MERGED DATASET")
    print("=" * 80)
    
    # Global
    print(f"\n📈 GLOBAL STATISTICS")
    print(f"   Total offers: {len(df):,}")
    
    # By source
    source_counts = df['source'].value_counts()
    print(f"\n📦 DISTRIBUTION BY SOURCE")
    for source, count in source_counts.items():
        pct = count / len(df) * 100
        print(f"   {source}: {count:,} offers ({pct:.1f}%)")

    # Global indicator: % offers with ROME code by source
    print(f"\n📌 ROME COVERAGE BY SOURCE")
    for source, count in source_counts.items():
        source_df = df[df['source'] == source]
        rome_count = source_df['rome_code'].notna() & (source_df['rome_code'] != '')
        rome_with_code = int(rome_count.sum())
        rome_pct = (rome_with_code / len(source_df) * 100) if len(source_df) > 0 else 0
        print(f"   {source}: {rome_with_code:,}/{len(source_df):,} offers with ROME ({rome_pct:.1f}%)")
    
    # ROME codes
    rome_stats = df[df['rome_code'].notna() & (df['rome_code'] != '')]
    print(f"\n🎯 ROME CODES")
    print(f"   Offers with ROME: {len(rome_stats):,} ({len(rome_stats)/len(df)*100:.1f}%)")
    print(f"   Unique ROME codes: {df['rome_code'].nunique()}")
    
    # Top 10 ROME codes
    print(f"\n🏆 TOP 10 ROME CODES")
    top_rome = df['rome_code'].value_counts().head(10)
    for rome, count in top_rome.items():
        pct = count / len(df) * 100
        label = df[df['rome_code'] == rome]['rome_label'].iloc[0] if not df[df['rome_code'] == rome].empty else " N/A"
        if(label is None or label.strip() == ""):
            label = "N/A"
        rome = rome if pd.notna(rome) and rome != '' else "N/A"
        print(f"   {rome}: {count:,} ({pct:.1f}%) - {label}")

    # ROME codes by source with label/description
    print(f"\n🧭 ROME CODES BY SOURCE")
    if 'source' in df.columns and 'rome_code' in df.columns:
        rome_by_source_df = df[
            df['source'].notna()
            & df['rome_code'].notna()
            & (df['rome_code'] != '')
        ].copy()

        if rome_by_source_df.empty:
            print("   No source/ROME data available")
        else:
            for source in rome_by_source_df['source'].dropna().unique():
                source_df = rome_by_source_df[rome_by_source_df['source'] == source]
                print(f"\n   Source: {source}")

                top_source_rome = source_df['rome_code'].value_counts().head(10)
                for rome, count in top_source_rome.items():
                    source_rome_rows = source_df[source_df['rome_code'] == rome]
                    label = "N/A"
                    if 'rome_label' in source_rome_rows.columns:
                        labels = source_rome_rows['rome_label'].dropna().astype(str)
                        labels = labels[labels.str.strip() != ""]
                        if not labels.empty:
                            label = labels.iloc[0]

                    pct_source = count / len(source_df) * 100
                    print(f"      {rome}: {count:,} ({pct_source:.1f}%) - {label}")
    else:
        print("   Missing required columns: source and/or rome_code")
    
    # Distribution of ROME codes
    rome_counts = df['rome_code'].value_counts()
    print(f"\n📊 DISTRIBUTION OF ROME CODES")
    print(f"   Codes with >1000 offers: {(rome_counts > 1000).sum()}")
    print(f"   Codes with 100-1000 offers: {((rome_counts >= 100) & (rome_counts <= 1000)).sum()}")
    print(f"   Codes with 50-100 offers: {((rome_counts >= 50) & (rome_counts < 100)).sum()}")
    print(f"   Codes with <50 offers: {(rome_counts < 50).sum()}")
    
    # Contract types
    print(f"\n📝 CONTRACT TYPES")
    contract_counts = df['contract_type'].value_counts().head(5)
    for contract, count in contract_counts.items():
        pct = count / len(df) * 100
        print(f"   {contract}: {count:,} ({pct:.1f}%)")
    
    # Experience levels
    print(f"\n💼 EXPERIENCE LEVELS")
    exp_counts = df['experience_level'].value_counts().head(5)
    for exp, count in exp_counts.items():
        pct = count / len(df) * 100
        print(f"   {exp}: {count:,} ({pct:.1f}%)")
    
    # Existing skills
    skills = df['skills'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
    print(f"\n🎓 SKILLS")
    print(f"   Offers with skills: {skills:,} ({skills/len(df)*100:.1f}%)")
    
    # Data quality
    print(f"\n✅ DATA QUALITY")
    
    def _calculate_fill_rate(series) -> tuple:
        """Calculate fill rate for a column (handles str, list, numeric types)"""
        total = len(series)
        if total == 0:
            return 0, 0.0
        
        # For string columns: not null + not empty after strip
        if series.dtype == 'object':
            # Check if it's a list column (first non-null value is a list)
            first_valid = series.dropna().head(1)
            if len(first_valid) > 0 and isinstance(first_valid.iloc[0], list):
                # List column: count non-empty lists
                filled = series.apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
            else:
                # String column: count non-null and non-empty after strip
                filled = (series.notna() & (series.astype(str).str.strip() != "")).sum()
        else:
            # Numeric/datetime: just count non-null
            filled = series.notna().sum()
        
        rate = (filled / total * 100) if total > 0 else 0.0
        return int(filled), rate
    
    # Key fields summary
    title_filled, title_rate = _calculate_fill_rate(df['title'])
    desc_filled, desc_rate = _calculate_fill_rate(df['description'])
    url_filled, url_rate = _calculate_fill_rate(df['url'])
    rome_filled, rome_rate = _calculate_fill_rate(df['rome_code'])

    print(f"   Title provided: {title_filled:,} ({title_rate:.1f}%)")
    print(f"   Description provided: {desc_filled:,} ({desc_rate:.1f}%)")
    print(f"   URL provided: {url_filled:,} ({url_rate:.1f}%)")
    print(f"   ROME code provided: {rome_filled:,} ({rome_rate:.1f}%)")
    
    # All fields fill rate analysis
    print(f"\n   📋 FILL RATE FOR ALL FIELDS (sorted by rate)")
    field_stats = []
    for col in df.columns:
        filled, rate = _calculate_fill_rate(df[col])
        field_stats.append((col, filled, rate))
    
    # Sort by fill rate descending
    field_stats_sorted = sorted(field_stats, key=lambda x: x[2], reverse=True)
    
    for col, filled, rate in field_stats_sorted:
        print(f"      {col:30s}: {filled:>8,}/{len(df):>8,} ({rate:>5.1f}%)")

    print(f"\n   🔎 DATA QUALITY BY SOURCE")
    if 'source' in df.columns:
        for source in df['source'].dropna().unique():
            source_df = df[df['source'] == source]
            if source_df.empty:
                continue

            source_total = len(source_df)
            source_title_filled, source_title_rate = _calculate_fill_rate(source_df['title'])
            source_desc_filled, source_desc_rate = _calculate_fill_rate(source_df['description'])
            source_url_filled, source_url_rate = _calculate_fill_rate(source_df['url'])
            source_rome_filled, source_rome_rate = _calculate_fill_rate(source_df['rome_code'])

            print(f"\n   Source: {source}")
            print(f"      Title provided: {source_title_filled:,}/{source_total:,} ({source_title_rate:.1f}%)")
            print(f"      Description provided: {source_desc_filled:,}/{source_total:,} ({source_desc_rate:.1f}%)")
            print(f"      URL provided: {source_url_filled:,}/{source_total:,} ({source_url_rate:.1f}%)")
            print(f"      ROME code provided: {source_rome_filled:,}/{source_total:,} ({source_rome_rate:.1f}%)")
    else:
        print("   Missing required column: source")

    # Random samples by source
    print(f"\n🎲 10 RANDOM ENTRIES BY SOURCE")
    if 'source' in df.columns:
        def _compact_text(value, max_len: int = 100) -> str:
            if value is None:
                return "N/A"

            is_na = pd.isna(value)
            if isinstance(is_na, bool) and is_na:
                return "N/A"

            text = str(value).strip().replace("\n", " ")
            if text == "":
                return "N/A"
            return text if len(text) <= max_len else text[: max_len - 3] + "..."

        for source in df['source'].dropna().unique():
            source_df = df[df['source'] == source]
            if source_df.empty:
                continue

            sample_size = min(10, len(source_df))
            sample_df = source_df.sample(n=sample_size)

            print(f"\n   Source: {source} | {sample_size} random entries")
            
            # Define field order for readability
            priority_fields = ["id", "source", "title", "rome_code", "rome_label", "contract_type", "url"]
            ordered_cols = [col for col in priority_fields if col in sample_df.columns]
            other_cols = [col for col in sample_df.columns if col not in priority_fields]
            all_cols = ordered_cols + other_cols
            
            for idx, (_, row) in enumerate(sample_df.iterrows(), start=1):
                row_id = _compact_text(row.get("id", "N/A"), 40)
                print(f"      [{idx:02d}] id={row_id}")

                for col in all_cols:
                    if col == "id":  # Already displayed in header
                        continue
                    value = _compact_text(row.get(col, "N/A"), 140)
                    print(f"           {col}: {value}")

                print("           " + "-" * 60)
    else:
        print("   Missing required column: source")
    
    print("\n" + "=" * 80 + "\n")
