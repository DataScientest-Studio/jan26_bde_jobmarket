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
import tqdm
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
# Reading France Travail (Bronze)
# =============================
def process_ft_file_with_timing(storage, key: str) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Manage jsonl reading with detailed performance measurements.
    Returns both the records and a dict of timings for each step:
    - network_latency: time to get_object and start reading
    - data_transfer: time to read the entire body
    - json_parsing: time to parse all lines into records
    - data_transform: time to transform the records
    - file_size_bytes: size of the file in bytes
    - record_count: number of records

    Return format: (records, timings):
    - records: list of parsed and transformed records
    - timings: dict with timing information for each step
    """
    records = []
    timings = {
        'network_latency': 0,
        'data_transfer': 0,
        'json_parsing': 0,
        'data_transform': 0,
        'file_size_bytes': 0,
        'record_count': 0
    }
    
    try:
        # Measure 1: Network latency (open connection + metadata)
        t0 = time.perf_counter()
        full_key = storage._full_key(key)
        resp = storage.client.get_object(Bucket=storage.bucket, Key=full_key)
        t1 = time.perf_counter()
        timings['network_latency'] = t1 - t0
        
        # Measure 2: Data transfer (read body)
        t2 = time.perf_counter()
        body_bytes = resp["Body"].read()
        t3 = time.perf_counter()
        timings['data_transfer'] = t3 - t2
        timings['file_size_bytes'] = len(body_bytes)
        
        # Measure 3: Parsing JSON
        t4 = time.perf_counter()
        parsed_records = []
        for line_bytes in body_bytes.split(b'\n'):
            if not line_bytes or line_bytes.isspace():
                continue
            try:
                if USE_ORJSON:
                    parsed_records.append(orjson.loads(line_bytes))
                else:
                    line = line_bytes.decode('utf-8', errors='replace').strip()
                    parsed_records.append(json.loads(line))
            except Exception:
                continue
        t5 = time.perf_counter()
        timings['json_parsing'] = t5 - t4
        timings['record_count'] = len(parsed_records)
        
        # Measure 4: Data transformation
        t6 = time.perf_counter()
        for record in parsed_records:
            try:
                # Secure extraction of origineOffre.urlOrigine
                url = ""
                origine_offre = record.get("origineOffre", {})
                if isinstance(origine_offre, dict):
                    url = origine_offre.get("urlOrigine", "")
                
                # Secure extraction of lieuTravail
                lieu_travail = record.get("lieuTravail", {})
                commune = ""
                code_postal_prefix = ""
                if isinstance(lieu_travail, dict):
                    commune = lieu_travail.get("commune", "")
                    code_postal = lieu_travail.get("codePostal", "")
                    code_postal_prefix = code_postal[:2] if code_postal else ""
                
                # Secure extraction of entreprise
                entreprise = record.get("entreprise", {})
                company_name = ""
                if isinstance(entreprise, dict):
                    company_name = entreprise.get("nom", "")
                
                # Secure extraction of salaire
                salaire = record.get("salaire", {})
                salary_text = ""
                if isinstance(salaire, dict):
                    salary_text = salaire.get("libelle", "")
                
                # Expected FT structure
                records.append({
                    "source": "FT",
                    "id": str(record.get("id", "")),
                    "intitule": normalize_text(record.get("intituleOffre", "")),
                    "description": normalize_text(record.get("description", "")),
                    "profile": normalize_text(record.get("experienceExigence", "")),
                    "rome_code": record.get("romeCode", ""),
                    "rome_label": record.get("romeLibelle", ""),
                    "contract_type": record.get("typeContrat", ""),
                    "experience_level": record.get("experienceLibelle", ""),
                    "salary_min": salary_text,
                    "salary_max": None,
                    "location_city": commune,
                    "location_department": code_postal_prefix,
                    "published_at": record.get("dateCreation", ""),
                    "updated_at": record.get("dateActualisation", ""),
                    "url": url,
                    "existing_skills": normalize_list_to_strings(record.get("competences", [])),
                    "company_name": company_name,
                })
            except Exception:
                continue
        t7 = time.perf_counter()
        timings['data_transform'] = t7 - t6
        
    except Exception as e:
        logger.warning(f"   ⚠️ Erreur lecture {key}: {e}")
    
    return records, timings


def process_ft_file(storage, key: str) -> List[Dict]:
    """
    Traite un fichier JSONL France Travail et retourne une liste de records.
    Fonction helper pour parallélisation.
    """
    records = []
    
    try:
        # Utilise read_jsonl optimisé du storage (lit tout d'un coup)
        # Au lieu de get_object + iter_lines (plus lent)
        for record in storage.read_jsonl(key):
            try:
                # record est déjà parsé par storage.read_jsonl()
                
                # Extraction sécurisée de origineOffre.urlOrigine
                url = ""
                origine_offre = record.get("origineOffre", {})
                if isinstance(origine_offre, dict):
                    url = origine_offre.get("urlOrigine", "")
                
                # Extraction sécurisée de lieuTravail
                lieu_travail = record.get("lieuTravail", {})
                commune = ""
                code_postal_prefix = ""
                if isinstance(lieu_travail, dict):
                    commune = lieu_travail.get("commune", "")
                    code_postal = lieu_travail.get("codePostal", "")
                    code_postal_prefix = code_postal[:2] if code_postal else ""
                
                # Extraction sécurisée de entreprise
                entreprise = record.get("entreprise", {})
                company_name = ""
                if isinstance(entreprise, dict):
                    company_name = entreprise.get("nom", "")
                
                # Extraction sécurisée de salaire
                salaire = record.get("salaire", {})
                salary_text = ""
                if isinstance(salaire, dict):
                    salary_text = salaire.get("libelle", "")
                
                # Structure FT attendue
                records.append({
                    "source": "FT",
                    "id": str(record.get("id", "")),
                    "intitule": normalize_text(record.get("intituleOffre", "")),
                    "description": normalize_text(record.get("description", "")),
                    "profile": normalize_text(record.get("experienceExigence", "")),
                    "rome_code": record.get("romeCode", ""),
                    "rome_label": record.get("romeLibelle", ""),
                    "contract_type": record.get("typeContrat", ""),
                    "experience_level": record.get("experienceLibelle", ""),
                    "salary_min": salary_text,
                    "salary_max": None,
                    "location_city": commune,
                    "location_department": code_postal_prefix,
                    "published_at": record.get("dateCreation", ""),
                    "updated_at": record.get("dateActualisation", ""),
                    "url": url,
                    "existing_skills": normalize_list_to_strings(record.get("competences", [])),
                    "company_name": company_name,
                })
                
            except Exception as e:
                # Erreur d'extraction sur un record spécifique
                continue
                
    except Exception as e:
        logger.warning(f"   ⚠️ Erreur lecture {key}: {e}")
    
    return records


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
    
    # 🔬 PROFILING : Mesurer sur les 300 premiers fichiers
    profiling_enabled = os.getenv("ENABLE_PROFILING", "true").lower() == "true"
    profiling_sample_size = min(300, total)
    
    if profiling_enabled and total > 0:
        logger.info(f"\n🔬 PROFILING activé : analyse de {profiling_sample_size} fichiers...")
        profiling_timings = []
        
        with ThreadPoolExecutor(max_workers=min(10, profiling_sample_size)) as executor:
            futures = {executor.submit(process_ft_file_with_timing, storage, key): key 
                      for key in keys[:profiling_sample_size]}
            
            for future in as_completed(futures):
                try:
                    records, timings = future.result()
                    all_records.extend(records)
                    profiling_timings.append(timings)
                    processed += 1
                except Exception as e:
                    logger.error(f"   ❌ Erreur profiling: {e}")
        
        # Calculer les statistiques
        if profiling_timings:
            avg_latency = sum(t['network_latency'] for t in profiling_timings) / len(profiling_timings)
            avg_transfer = sum(t['data_transfer'] for t in profiling_timings) / len(profiling_timings)
            avg_parsing = sum(t['json_parsing'] for t in profiling_timings) / len(profiling_timings)
            avg_transform = sum(t['data_transform'] for t in profiling_timings) / len(profiling_timings)
            avg_size = sum(t['file_size_bytes'] for t in profiling_timings) / len(profiling_timings)
            avg_records = sum(t['record_count'] for t in profiling_timings) / len(profiling_timings)
            
            total_per_file = avg_latency + avg_transfer + avg_parsing + avg_transform
            
            # Projections sur tous les fichiers
            proj_latency = avg_latency * total
            proj_transfer = avg_transfer * total
            proj_parsing = avg_parsing * total
            proj_transform = avg_transform * total
            proj_total = total_per_file * total
            
            logger.info(f"""
📊 RÉSULTATS PROFILING ({profiling_sample_size} fichiers):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Moyennes par fichier:
    • Latence réseau    : {avg_latency*1000:>7.1f} ms  ({avg_latency/total_per_file*100:>5.1f}%)
    • Transfert données : {avg_transfer*1000:>7.1f} ms  ({avg_transfer/total_per_file*100:>5.1f}%)
    • Parsing JSON      : {avg_parsing*1000:>7.1f} ms  ({avg_parsing/total_per_file*100:>5.1f}%)
    • Transformation    : {avg_transform*1000:>7.1f} ms  ({avg_transform/total_per_file*100:>5.1f}%)
    ────────────────────────────────────────────────────
    • TOTAL par fichier : {total_per_file*1000:>7.1f} ms  (100.0%)
    • Taille moyenne    : {avg_size/1024:>7.1f} KB
    • Records moyens    : {avg_records:>7.1f}

  Projections pour {total} fichiers:
    • Latence réseau    : {proj_latency:>7.1f} s  ({proj_latency/60:>6.2f} min)
    • Transfert données : {proj_transfer:>7.1f} s  ({proj_transfer/60:>6.2f} min)
    • Parsing JSON      : {proj_parsing:>7.1f} s  ({proj_parsing/60:>6.2f} min)
    • Transformation    : {proj_transform:>7.1f} s  ({proj_transform/60:>6.2f} min)
    ────────────────────────────────────────────────────
    • TOTAL estimé      : {proj_total:>7.1f} s  ({proj_total/60:>6.2f} min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Recommandation:
   {"La latence réseau domine (>50%). Consolidez en gros fichiers !" if avg_latency/total_per_file > 0.5 else "Optimisations JSON parser et transformations recommandées."}
""")
        
        # Ajuster les clés restantes
        keys = keys[profiling_sample_size:]
        total = len(keys)
        logger.info(f"📦 Suite du traitement : {total} fichiers restants...\n")
    
    if use_batching and total > 0:
        # Traitement par batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_keys = keys[batch_start:batch_end]
            
            logger.info(f"   📦 Batch {batch_start//batch_size + 1}: fichiers {batch_start+1}-{batch_end}/{total}")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_ft_file, storage, key): key for key in batch_keys}
                
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
            futures = {executor.submit(process_ft_file, storage, key): key for key in keys}
            
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

    bar = tqdm.tqdm(df.iterrows(), total=len(df), desc="   Normalizing FT data", mininterval=0.5)
    for _, row in bar:

        bar.set_postfix({"id": row.get("id", "N/A")})

        silver_obj = Silver_Datamodel(
            source="FT",
            id=str(row.get("id", "")),
            title=normalize_text(row.get("intitule", "")),
            description=normalize_text(row.get("description", "")),
            profile=normalize_text(row.get("profile", "")),
            rome_code=row.get("rome_code", ""),
            rome_label=row.get("rome_label", ""),
            contract_type=row.get("contract_type", ""),
            experience_level=row.get("experience_level", ""),
            salary_min=row.get("salary_min"),
            salary_max=row.get("salary_max"),
            job_city=row.get("location_city", ""),
            location_department=row.get("location_department", ""),
            published_at=row.get("published_at", ""),
            updated_at=row.get("updated_at", ""),
            url=row.get("url", ""),
            skills=row.get("existing_skills", []),
            company_name=row.get("company_name", "")
        )
        normalized_data.append(silver_obj.to_dict())
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f"   Normalization completed in {elapsed_time:.2f} seconds")
    
    return pd.DataFrame(normalized_data)    

# =============================
# Reading WTTJ (Silver)
# =============================
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
        
        # Parse JSON strings if needed
        skills = normalize_list_to_strings(row.get("skills", []))
        
        # Use Silver_Datamodel for normalization
        silver_obj = Silver_Datamodel(
            source="WTTJ",
            id=str(row.get("wttj_reference", "")),
            title=normalize_text(row.get("name", "")),
            description=normalize_text(row.get("description", "")),
            profile=normalize_text(row.get("profile", "")),
            rome_code=row.get("rome_code", ""),
            rome_label=row.get("rome_label", ""),
            contract_type=row.get("contract_type", ""),
            experience_level=row.get("experience_level", ""),
            salary_min=row.get("salary_min"),
            salary_max=row.get("salary_max"),
            job_city="",  # À extraire de offices si besoin
            location_department="",
            published_at=row.get("published_at", ""),
            updated_at=row.get("updated_at", ""),
            url=row.get("canonical_url", ""),
            skills=skills,
            company_name=""  # Disponible dans company_summary si besoin
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
# Statistics
# =============================
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
        label = df[df['rome_code'] == rome]['rome_label'].iloc[0] if not df[df['rome_code'] == rome].empty else ""
        print(f"   {rome}: {count:,} ({pct:.1f}%) - {label}")
    
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
    skills_present = df['existing_skills'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
    print(f"\n🎓 EXISTING SKILLS")
    print(f"   Offers with skills: {skills_present:,} ({skills_present/len(df)*100:.1f}%)")
    
    # Data quality
    print(f"\n✅ DATA QUALITY")
    print(f"   Title provided: {df['intitule'].notna().sum():,} ({df['intitule'].notna().sum()/len(df)*100:.1f}%)")
    print(f"   Description provided: {df['description'].notna().sum():,} ({df['description'].notna().sum()/len(df)*100:.1f}%)")
    print(f"   URL provided: {df['url'].notna().sum():,} ({df['url'].notna().sum()/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 80 + "\n")


# =============================
# Save dataset
# =============================
def save_dataset(df: pd.DataFrame, storage, format: str = "parquet") -> str:
    """
    Save the merged dataset.
    
    Args:
        df: DataFrame to save
        storage: Storage instance
        format: Output format (parquet, jsonl, csv)
        
    Returns:
        Output key where data was saved
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "parquet":
        output_key = f"merged_dataset_{timestamp}.parquet"
        logger.info(f"💾 Sauvegarde Parquet: {output_key}")
        storage.write_parquet(output_key, df)
        
    elif format == "jsonl":
        output_key = f"merged_dataset_{timestamp}.jsonl"
        logger.info(f"💾 Save JSONL: {output_key}")
        
        # Convert to JSONL
        jsonl_data = df.to_json(orient='records', lines=True, force_ascii=False)
        storage.put_object(output_key, jsonl_data.encode('utf-8'))
        
    elif format == "csv":
        output_key = f"merged_dataset_{timestamp}.csv"
        logger.info(f"💾 Save CSV: {output_key}")
        
        # Convert lists to JSON strings for CSV
        df_csv = df.copy()
        df_csv['existing_skills'] = df_csv['existing_skills'].apply(lambda x: json.dumps(x) if isinstance(x, list) else "[]")
        
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
    storage_wttj = get_storage_from_env("bronze", "welcometothejungle")
    
    # Auto-detection of prefixes if not specified
    if not ft_prefix:
        ft_prefix = os.getenv("FT_BRONZE_PREFIX")
        if not ft_prefix:
            logger.info("🔍 Auto-detection of FT Bronze prefix...")
            if progress_callback:
                progress_callback("detecting_ft_prefix", "Détection du préfixe FT...")
            ft_prefix = find_latest_data_prefix(storage_ft, "bronze", "offers")
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
            wttj_prefix = find_latest_data_prefix(storage_wttj, "silver", "")
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
   
    # Read FT Bronze JSONL files to a DataFrame with parallelization
    # and normalization to match Silver_Datamodel structure
    if progress_callback:
        progress_callback("reading_ft", "Lecture des données France Travail...")
    df_ft = read_ft_bronze_data(storage_ft, ft_prefix)
    
    if progress_callback:
        progress_callback("normalizing_ft", f"Normalisation FT ({len(df_ft):,} offres)...")
    df_ft = normalize_ft_data(df_ft)

    # Read Wttj Silver parquet file to a DataFrame
    # Normalize WTTJ data to match Silver_Datamodel structure
    if progress_callback:
        progress_callback("reading_wttj", "Lecture des données WTTJ...")
    df_wttj = read_wttj_parquet_file_to_df(storage_wttj, wttj_prefix)
    
    if progress_callback:
        progress_callback("normalizing_wttj", f"Normalisation WTTJ ({len(df_wttj):,} offres)...")
    df_wttj = normalize_wttj_data(df_wttj)

    # Merge and deduplicate datasets
    if progress_callback:
        progress_callback("merging", "Fusion et déduplication...")
    df_merged = merge_and_deduplicate(df_ft, df_wttj)
    
    # Statistics
    if progress_callback:
        progress_callback("computing_stats", "Calcul des statistiques...")
    print_statistics(df_merged)
    
    # Save merged dataset 
    if progress_callback:
        progress_callback("saving", f"Sauvegarde du dataset ({len(df_merged):,} offres)...")
    storage_output = get_storage_from_env("silver", "merged")

    output_key = save_dataset(df_merged, storage_output, output_format)
    
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
    
    merge_ft_wttj_datasets(
        ft_prefix=args.ft_prefix,
        wttj_prefix=args.wttj_prefix,
        output_prefix=args.output_prefix,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
