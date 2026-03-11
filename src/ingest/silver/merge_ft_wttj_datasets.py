"""
============================================
Fusion des datasets FT et WTTJ avec ROME
============================================

Fusionne les données de France Travail (Bronze) et Welcome To The Jungle (Silver)
avec les codes ROME prédits pour créer un dataset d'entraînement unifié.

Usage:
    python -m src.data.make_merge_dataset_ft_wttj_with_rome
    python -m src.data.make_merge_dataset_ft_wttj_with_rome --format parquet
    python -m src.data.make_merge_dataset_ft_wttj_with_rome --output-prefix datasets/custom
"""
import argparse
import importlib.util
import json
import logging
import os
import re
import tqdm
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import orjson  # 10x faster than standard json
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False
    
import pandas as pd
from dotenv import load_dotenv

from src.storage.storage import get_storage_from_env
from src.ingest.data_models.silver_datamodel_class import Silver_Datamodel
from src.utils import find_latest_data_prefix, clean_html, normalize_text, normalize_list_to_strings
import src.utils.merge_dataset_utils as merge_utils

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================
# Configuration
# =============================
FT_BRONZE_PREFIX = os.getenv("FT_BRONZE_PREFIX", "offers/dt=2026-02-18/run_id=20260217T181105Z/segment=jobs_raw")
WTTJ_SILVER_PREFIX = os.getenv("WTTJ_SILVER_PREFIX", "silver/dt=2026-02-22/run_id=20260222T010000Z/segment=jobs")
MERGED_DATASET_PREFIX  = os.getenv("MERGED_DATASET_PREFIX", "datasets/ft_wttj_merged")


# =============================
# Type Conversion Helpers
# =============================
def safe_str_to_float(value: Any) -> float:
    """Convert various types to float, return 0.0 for invalid/None values."""
    if value is None or value == "":
        return 0.0
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return 0.0
    return 0.0


def safe_str_to_datetime(value: Any) -> Optional[datetime]:
    """Convert various types to datetime, return None for invalid/empty values."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        # Try common date formats (ISO 8601 with Z and microseconds, standard formats)
        for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y"]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        # Log warning if no format matched
        logger.warning(f"⚠️ Failed to parse datetime value: '{value}' (tried all known formats)")
        return None
    return None


def safe_str_to_str(value: Any) -> str:
    """Convert various types to string, return empty string for None values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


# =============================
# File Processing Helpers
# =============================
def _read_source_file(storage, key: str, mode: str) -> Any:
    """
    Generic helper to process a file depending on mode.

    Modes:
    - jsonl_records: returns List[Dict]
    - parquet_df: returns pd.DataFrame
    """
    if mode == "jsonl_records":
        records: List[Dict] = []
        try:
            for record in storage.read_jsonl(key):
                try:
                    records.append(record)
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"   ⚠️ Erreur lecture {key}: {e}")
        return records

    if mode == "parquet_df":
        try:
            return storage.read_parquet(key)
        except Exception as e:
            logger.error(f"   ❌ Erreur lecture {key}: {e}")
            return pd.DataFrame()

    raise ValueError(f"Unsupported processing mode: {mode}")

# =============================
# Reading France Travail (Bronze)
# =============================
def read_ft_bronze_data(storage, prefix: str) -> pd.DataFrame:
    """
    Lit les données France Travail depuis la couche Bronze (JSONL) avec parallélisation.
    
    Args:
        storage: Instance de storage
        prefix: Préfixe S3/local des données Bronze FT
        
    Returns:
        DataFrame avec les colonnes normalisées
    """
    logger.info(f"📂 Lecture France Travail Bronze: {prefix}")
    
    keys = list(storage.list_keys(prefix))
    logger.info(f"   Trouvé {len(keys)} fichiers JSONL")
    
    if len(keys) == 0:
        logger.warning(f"⚠️ Aucun fichier trouvé dans {prefix}")
        return pd.DataFrame()
    
    # Parallélisation avec ThreadPoolExecutor
    # Connection pool configuré à 50 connexions (S3_MAX_POOL_CONNECTIONS)
    # Augmentation à 40 workers (80% du pool) pour maximiser le throughput
    max_workers = int(os.getenv("FT_READ_WORKERS", "40"))
    
    # Traitement par batches pour gérer efficacement les gros volumes
    batch_size = max_workers * 12  # Traiter par batches de ~480 fichiers (40 workers * 12)
    use_batching = len(keys) > batch_size
    
    if use_batching:
        logger.info(f"   Mode batch activé: {max_workers} workers, {batch_size} fichiers/batch")
    else:
        logger.info(f"   Mode parallèle: {max_workers} workers")
    
    all_records = []
    processed = 0
    total = len(keys)
    
    if use_batching and total > 0:
        # Traitement par batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_keys = keys[batch_start:batch_end]
            
            logger.info(f"   📦 Batch {batch_start//batch_size + 1}: fichiers {batch_start+1}-{batch_end}/{total}")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_read_source_file, storage, key, mode="jsonl_records"): key for key in batch_keys}
                
                for future in as_completed(futures):
                    processed += 1
                    
                    try:
                        records = future.result()
                        all_records.extend(records)
                        
                        # Log de progression tous les 100 fichiers
                        if processed % 100 == 0 or processed == total:
                            rate = len(all_records) / processed if processed > 0 else 0
                            logger.info(f"   Progression: {processed}/{total} fichiers ({len(all_records):,} records, ~{rate:.1f} records/fichier)")
                            
                    except Exception as e:
                        key = futures[future]
                        logger.error(f"   ❌ Erreur traitement {key}: {e}")
    else:
        # Traitement direct si peu de fichiers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_read_source_file, storage, key, mode="jsonl_records"): key for key in keys}
            
            for future in as_completed(futures):
                processed += 1
                
                try:
                    records = future.result()
                    all_records.extend(records)
                    
                    if processed % 100 == 0 or processed == total:
                        rate = len(all_records) / processed if processed > 0 else 0
                        logger.info(f"   Progression: {processed}/{total} fichiers ({len(all_records):,} records, ~{rate:.1f} records/fichier)")
                        
                except Exception as e:
                    key = futures[future]
                    logger.error(f"   ❌ Erreur traitement {key}: {e}")
    
    # Créer DataFrame avec colonnes par défaut si vide
    if not all_records:
        logger.warning("⚠️ Aucune donnée FT trouvée, DataFrame vide créé")
        df = pd.DataFrame(columns=Silver_Datamodel.get_dataframe_columns())
    else:
        df = pd.DataFrame(all_records)
    
    logger.info(f"✅ France Travail: {len(df):,} offres chargées")
    
    # Debug: afficher les premières colonnes
    if len(df) > 0:
        logger.info(f"   Colonnes créées: {list(df.columns)}")
        logger.info(f"   Exemple URL: {df['url'].iloc[0] if 'url' in df.columns and len(df) > 0 else 'N/A'}")
    
    return df

def normalize_ft_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes FT data to match the Silver_Datamodel structure.
    """
    start_time = time.perf_counter()

    if len(df) == 0:
        logger.warning("⚠️ DataFrame FT empty, returning empty normalized DataFrame")
        return pd.DataFrame(columns=Silver_Datamodel.get_dataframe_columns())
    normalized_data = []

    total_rows = len(df)
    bar = tqdm.tqdm(df.iterrows(), total=total_rows, desc="   Normalizing FT data", mininterval=0.5)
    for processed_rows, (_, row) in enumerate(bar, 1):
        elapsed = time.perf_counter() - start_time
        remaining = ((total_rows - processed_rows) * (elapsed / processed_rows)) if processed_rows > 0 else 0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(max(remaining, 0)))

        bar.set_postfix({"id": row.get("id", "N/A"), "eta": eta_str})

        url = ""
        origine_offre = row.get("origineOffre", {})
        if isinstance(origine_offre, dict):
            url = origine_offre.get("urlOrigine", "")
        
        # Extraction sécurisée de lieuTravail
        lieu_travail = row.get("lieuTravail", {})
        commune = ""
        code_postal_prefix = ""
        job_city = ""
        job_postal_code = ""
        if isinstance(lieu_travail, dict):
            commune = lieu_travail.get("commune", "")
            job_postal_code = lieu_travail.get("codePostal", "")
            job_city = lieu_travail.get("libelle", "")
            code_postal_prefix = job_postal_code[:2] if job_postal_code else ""
        
        # Extraction sécurisée de entreprise
        entreprise = row.get("entreprise", {})
        company_name = ""
        if isinstance(entreprise, dict):
            company_name = entreprise.get("nom", "")
        
        # Extraction sécurisée de salaire
        salaire = row.get("salaire", {})
        salary_text = ""
        if isinstance(salaire, dict):
            salary_text = salaire.get("libelle", "")

        silver_obj = Silver_Datamodel(
            id=str(row.get("id", "")),
            source="FT",
            url=url,
            title=normalize_text(row.get("intitule", "")),
            description=normalize_text(row.get("description", "")),
            published_at=safe_str_to_datetime(row.get("dateCreation", "")),
            updated_at=safe_str_to_datetime(row.get("dateActualisation", "")),
            status="published",
            rome_code=row.get("romeCode", ""),
            rome_label=row.get("romeLibelle", ""),
            title_description=normalize_text(row.get("appellationlibelle", "")),
            contract_type=row.get("typeContratLibelle", ""),
            worktime="" if pd.isna(row.get("dureeTravailLibelleConverti")) else row.get("dureeTravailLibelleConverti", ""), 
            experience_level=row.get("experienceExige", ""),
            experience_description=row.get("experienceLibelle", ""), 
            naf_code=row.get("codeNAF", ""),
            job_city=job_city,
            job_postal_code=job_postal_code,
            company_name=company_name,
            company_city=commune,
            company_postal_code=job_postal_code,
            company_url=row.get("entreprise_url", ""),
            salary_min=safe_str_to_float(salary_text),
            salary_max=safe_str_to_float(salary_text),
            profile=normalize_text(row.get("profile", "")),
            location_department=code_postal_prefix,
            skills=[],
            salary_periodicity=safe_str_to_str(salary_text)
        )
        normalized_data.append(silver_obj.to_dict())
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f"   Normalization completed in {elapsed_time:.2f} seconds")
    
    return pd.DataFrame(normalized_data)    

# =============================
# Reading WTTJ (Silver)
# =============================
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
            futures = {executor.submit(_read_source_file, storage, key, mode="parquet_df"): key for key in parquet_keys}
            
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
            df = _read_source_file(storage, parquet_keys[0], mode="parquet_df")
            dfs.append(df)
        except Exception as e:
            logger.error(f"   ❌ Error reading {parquet_keys[0]}: {e}")
    
    if not dfs:
        logger.warning("⚠️ No WTTJ data found")
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"✅ Welcome To The Jungle: {len(df):,} offers loaded")
    
    # Normalize list-type fields to prevent PyArrow struct/non-struct mixing
    # (skills, offices, key_missions, sectors can contain mixed types across records)
    if len(df) > 0:
        for list_field in ['skills', 'offices', 'key_missions', 'sectors']:
            if list_field in df.columns:
                # Convert all rows to consistent List[str] format
                df[list_field] = df[list_field].apply(lambda x: normalize_list_to_strings(x) if x is not None else [])
                logger.info(f"   ✓ Normalized {list_field} to List[str]")
    
    return df    


def normalize_wttj_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize WTTJ data to match the Silver_Datamodel structure."""

    start_time = time.perf_counter()

    # Normalize WTTJ data to match FT structure
    if len(df) == 0:
        logger.warning("⚠️ DataFrame WTTJ empty, returning empty normalized DataFrame")
        return pd.DataFrame(columns=Silver_Datamodel.get_dataframe_columns())
    
    normalized_data = []
    # No logger.info here - tqdm desc will show the status
    bar = tqdm.tqdm(df.iterrows(), total=len(df), desc="   Normalizing WTTJ data", mininterval=0.5)
    
    for _, row in bar:
        bar.set_postfix({"id": row.get("wttj_reference", "N/A")})
        
        # Normalize list fields to consistent List[str] format
        skills = normalize_list_to_strings(row.get("skills", []))
        
        # Use Silver_Datamodel for normalization
        silver_obj = Silver_Datamodel(
            id=str(row.get("wttj_reference", "")),
            source="WTTJ",
            url=row.get("canonical_url", ""),
            title=normalize_text(row.get("name", "")),
            description=normalize_text(row.get("description", "")),
            published_at=safe_str_to_datetime(row.get("published_at", "")),
            updated_at=safe_str_to_datetime(row.get("updated_at", "")),
            status="published", 
            rome_code=row.get("rome_code", ""),
            rome_label=row.get("rome_label", ""),
            title_description=normalize_text(row.get("appellationlibelle", "")),
            contract_type=row.get("contract_type", ""),
            worktime="",
            experience_level=row.get("experience_level", ""),
            experience_description="",
            naf_code="",
            job_city="",
            job_postal_code="",
            company_name="",
            company_city="",
            company_postal_code="",
            company_url="",
            salary_min=safe_str_to_float(row.get("salary_min")),
            salary_max=safe_str_to_float(row.get("salary_max")),
            profile=normalize_text(row.get("profile", "")),
            location_department="",
            skills=[],
            salary_periodicity=""
        )
        
        # Convert to dict for DataFrame
        normalized_data.append(silver_obj.to_dict())
    
    # Create DataFrame with default columns if empty
    if not normalized_data:
        logger.warning("⚠️ No WTTJ data normalized")
        df_normalized = pd.DataFrame(columns=Silver_Datamodel.get_dataframe_columns())
    else:
        df_normalized = pd.DataFrame(normalized_data)
    
    # Debug: afficher les premières colonnes
    if len(df_normalized) > 0:
        logger.info(f"   Colonnes créées: {list(df_normalized.columns)}")
        logger.info(f"   Exemple URL: {df_normalized['url'].iloc[0] if 'url' in df_normalized.columns and len(df_normalized) > 0 else 'N/A'}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f"   Normalization completed in {elapsed_time:.2f} seconds")

    return df_normalized

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
        dt_ft: Date from FT Bronze prefix (for filename)
        dt_wttj: Date from WTTJ Silver prefix (for filename)
        
    Returns:
        Output key where data was saved
    """
    dt_current = datetime.now().strftime("%Y-%m-%d")

    if format == "parquet":
        output_key = f"merged_dt={dt_current}_ft_dt={dt_ft}_wttj_dt={dt_wttj}.parquet"
        logger.info(f"💾 Sauvegarde Parquet: {output_key}")
        storage.write_parquet(output_key, df)
        
    elif format == "jsonl":
        output_key = f"merged_dt={dt_current}_ft_dt={dt_ft}_wttj_dt={dt_wttj}.jsonl"
        logger.info(f"💾 Save JSONL: {output_key}")
        
        # Convert to JSONL
        jsonl_data = df.to_json(orient='records', lines=True, force_ascii=False)
        storage.put_object(output_key, jsonl_data.encode('utf-8'))
        
    elif format == "csv":
        output_key = f"merged_dt={dt_current}_ft_dt={dt_ft}_wttj_dt={dt_wttj}.csv"
        logger.info(f"💾 Save CSV: {output_key}")
        
        # Convert lists to JSON strings for CSV
        df_csv = df.copy()
        #df_csv['existing_skills'] = df_csv['existing_skills'].apply(lambda x: json.dumps(x) if isinstance(x, list) else "[]")
        
        csv_data = df_csv.to_csv(index=False)
        storage.put_object(output_key, csv_data.encode('utf-8'))
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use parquet, jsonl, or csv.")
    
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
    progress_callback: Optional[callable] = None,
) -> Dict:
    """
    Main function to merge datasets.
    
    Args:
        ft_prefix: FT Bronze data prefix
        wttj_prefix: WTTJ Silver data prefix
        output_prefix: Output prefix
        output_format: Output format
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dict with success status, stats, and output key
    """
    start_time = time.perf_counter()
    # Storages
    storage_ft = get_storage_from_env("bronze", "france_travail")
    # For WTTJ, we read from Silver to get the latest normalized data with ROME codes
    storage_wttj = get_storage_from_env("silver", "welcometothejungle")
    
    # Auto-detection of prefixes if not specified
    if not ft_prefix:
        ft_prefix = os.getenv("FT_BRONZE_PREFIX")
        if not ft_prefix:
            logger.info("🔍 Auto-detection of FT Bronze prefix...")
            if progress_callback:
                progress_callback("detecting_ft_prefix", "Détection du préfixe FT...")
            ft_prefix = find_latest_data_prefix(storage_ft, "offers")
            if ft_prefix:
                logger.info(f"   ✓ Found: {ft_prefix}")
            else:
                logger.warning("   ⚠️ No FT Bronze data found")
                ft_prefix = FT_BRONZE_PREFIX  # Fallback to default value
    
    if not wttj_prefix:
        wttj_prefix = os.getenv("WTTJ_SILVER_PREFIX")
        if not wttj_prefix:
            logger.info("🔍 Auto-detection of WTTJ Silver prefix...")
            if progress_callback:
                progress_callback("detecting_wttj_prefix", "Détection du préfixe WTTJ...")
            wttj_prefix = find_latest_data_prefix(storage_wttj, "", "")
            wttj_prefix+= "/segment=jobs"  # We want to ensure we get the jobs segment
            if wttj_prefix:
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

    # Read Wttj Silver parquet file to a DataFrame
    # Normalize WTTJ data to match Silver_Datamodel structure
    if progress_callback:
        progress_callback("reading_wttj", "Lecture des données WTTJ...")
    df_wttj = read_wttj_parquet_file_to_df(storage_wttj, wttj_prefix)
    
    if progress_callback:
        progress_callback("normalizing_wttj", f"Normalisation WTTJ ({len(df_wttj):,} offres)...")
    df_wttj = normalize_wttj_data(df_wttj)

    # Read FT Bronze JSONL files to a DataFrame with parallelization
    # and normalization to match Silver_Datamodel structure
    if progress_callback:
        progress_callback("reading_ft", "Lecture des données France Travail...")
    df_ft = read_ft_bronze_data(storage_ft, ft_prefix)
    
    if progress_callback:
        progress_callback("normalizing_ft", f"Normalisation FT ({len(df_ft):,} offres)...")
    df_ft = normalize_ft_data(df_ft)


    # Merge and deduplicate datasets
    if progress_callback:
        progress_callback("merging", "Fusion et déduplication...")
    df_merged = merge_and_deduplicate(df_ft, df_wttj)

    # =======================================
    # Apply normalization to mergeded dataset
    # =======================================
    df_merged = merge_utils.normalize_contracts(df_merged, merge_utils.PATTERNS_CONTRACT_NORMALIZE)
    # Create composite experience column for normalization
    # TODO : change column name in source
    df_merged['experience_source_composite'] = df_merged.apply(merge_utils.get_experience_col, axis=1)

    # Normalize experience levels using the composite column
    df_merged = merge_utils.normalize_experience(df_merged,'experience_source_composite')
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
    merge_utils.print_statistics(df_merged)   
    
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


def main():
    """Entry point for CLI"""
    parser = argparse.ArgumentParser(description="Merge FT + WTTJ datasets with ROME codes")
    parser.add_argument("--ft-prefix", help="FT Bronze data prefix")
    parser.add_argument("--wttj-prefix", help="WTTJ Silver data prefix")
    parser.add_argument("--output-prefix", help="Output prefix")
    parser.add_argument("--format", choices=["parquet", "jsonl", "csv"], default="parquet", help="Output format")
    
    args = parser.parse_args()

    wttj_prefix=args.wttj_prefix
    #wttj_prefix= "dt=2026-02-28/segment=jobs"

    merge_ft_wttj_datasets(
        ft_prefix=args.ft_prefix,
        wttj_prefix=wttj_prefix,
        output_prefix=args.output_prefix,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
