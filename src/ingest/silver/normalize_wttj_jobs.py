"""Normalize WTTJ bronze offers into canonical silver records.

Responsibilities of this file:
- Source-specific normalization for Welcome to the Jungle (bronze -> silver).
- JSONL parsing, source field extraction, text/list cleanup, and type harmonization.
- WTTJ-specific enrichment (ROME prediction) during source normalization.
- Output of canonical `Silver_Datamodel`-compatible rows under `dt=.../segment=jobs`.

Out of scope for this file:
- Cross-source deduplication (FT vs WTTJ).
- Cross-source harmonization rules for merged datasets.
"""
from __future__ import annotations
import os
import uuid
import time
import pandas as pd
import json
from typing import List

from src.config.env import load_project_env
load_project_env()  # safe à rappeler (idempotent)

# Classes
from src.ingest.data_models.silver_datamodel_class import NormalizeResult, Silver_Datamodel
# Utils
from src.utils.wttj_utils import get_field_or_default, get_json_field_from_record, _fix_double_encoded_dict
from src.utils.text_processing import clean_html, normalize_list_to_strings, normalize_text
from src.utils.normalization_helpers import safe_str_to_datetime, safe_str_to_float, write_dataframe_with_format
from src.utils.storage_tools import get_last_dt_from_storage
from src.utils.log_to_db import log_to_db
from src.utils.rome import get_rome_code_from_ml_prediction_via_api
from src.storage.storage import get_storage_from_env
import src.utils.time_helpers as time_helpers

# Pour forcer l’affichage en console pour tous les loggers,
# à ajouter avant toute création de logger :
import logging
logging.basicConfig(level=logging.INFO)
# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger(__name__)
structured_logger = logging.getLogger("structured")


def _normalize_profession_value(value):
    """
    Normalise "profession" field to ensure it's a string. If it's a dict with a "name" key, use that.
    Otherwise, it generate mixed types in the profession column which can cause issues 
    in downstream processing (parquet schema saved with mixed types, or jsonl with inconsistent formats).
    """
    if value is None:
        return None
    if isinstance(value, dict):
        name = value.get("name")
        if name is not None:
            return str(name)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def normalize_wttj_jobs(dt: str, output_format: str = "parquet") -> NormalizeResult:
    """ 
    Normalize WTTJ jobs for a given date (dt) and save to silver.
    Args:        
    - dt: Date to process (format YYYYMMDD). If "latest" or empty, processes the latest date available in bronze.
    - output_format: Format to save the normalized data (parquet or jsonl).    

    """
    job_id = f"wttj-normalize-{dt}-{uuid.uuid4().hex[:8]}"
    status = "RUNNING"
    files: List[str] = []
    errors = 0

    # Strorage instances
    storage_bronze = get_storage_from_env("bronze", "welcometothejungle")
    storage_silver = get_storage_from_env("silver", "welcometothejungle")

    # Optimized: list only runid folders, then segment=jobs_raw keys per runid
    if( dt is None or dt == "" or dt == "latest"):
        dt = get_last_dt_from_storage(storage_bronze, "")

    # Efficient key discovery:
    # 1) list run_id folders under dt
    # 2) inside each run_id, list only segment=jobs_raw
    # 3) list JSONL files only in those segments
    dt_prefix = f"dt={dt}/"
    start_time = time.time()
    jobs_raw_keys = []


    runid_folders = storage_bronze.list_prefixes(dt_prefix)
    for runid_folder in runid_folders:
        if not (runid_folder.startswith("run_id=") or runid_folder.startswith("runid=")):
            continue

        runid_prefix = dt_prefix + runid_folder
        segment_folders = storage_bronze.list_prefixes(runid_prefix)
        for segment_folder in segment_folders:
            if not segment_folder.startswith("segment=jobs_raw"):
                continue

            segment_prefix = runid_prefix + segment_folder
            segment_keys = storage_bronze.list_keys(segment_prefix)
            jobs_raw_keys.extend([k for k in segment_keys if k.endswith(".jsonl")])
    try:
        log_to_db(
            endpoint="normalize_wttj_jobs",
            level="INFO",
            message=f"Start job: {job_id}, dt={dt}, output_format={output_format}, files={len(jobs_raw_keys)}",
            job_id=job_id,
            dt=dt,
            output_format=output_format,
            files=len(jobs_raw_keys),
            status="RUNNING"
        )
    except Exception as e:
        logger.warning(f"[normalize_wttj_jobs] log_to_db start failed: {e}")

    logger.info(f"[normalize_wttj_jobs] Start | dt={dt} | output_format={output_format} | job_id={job_id} | {len(jobs_raw_keys)} files to process")

    # Estimation dynamique du nombre total de records
    total_files = len(jobs_raw_keys)
    
    # Initialize data list for all files
    data = []
    
    # Process each file
    file_counter = 0
    global_start_time = time.time()
    rome_api_calls = 0
    rome_db_log_every = int(os.getenv("WTTJ_ROME_DB_LOG_EVERY", "100"))

    for i, key in enumerate(jobs_raw_keys, 1):
        elapsed_global = time.time() - global_start_time
        processed_before_current = i - 1
        global_remaining = (total_files - processed_before_current) * (elapsed_global / processed_before_current) if processed_before_current > 0 else 0
        global_eta_str = time_helpers.format_eta(global_remaining)
        logger.info(f"[normalize_wttj_jobs] File {i}/{total_files} | key={key} | Global ETA: {global_eta_str}")
        try:
            obj = storage_bronze.get_object_jsonl(key=key)
            # Iterlines to preserve memory load
            # No tqdm to avoid to load entire file in memory
            line_count = 0
            file_start_time = time.time()
            content_length = obj.get("ContentLength") if isinstance(obj, dict) else None
            bytes_processed = 0

            for line_bytes in obj["Body"].iter_lines():                
                #print(line_bytes[:3000])
                line_count += 1
                bytes_processed += len(line_bytes) + 1
                if line_count % 500 == 0:
                    elapsed_file = time.time() - file_start_time
                    # Estimate global ETA continuously using current file progress (bytes-based when available).
                    elapsed_global_now = time.time() - global_start_time
                    if content_length and content_length > 0:
                        current_file_progress = min(bytes_processed / content_length, 1.0)
                    else:
                        current_file_progress = 0.0
                    processed_equivalent_files = processed_before_current + current_file_progress
                    if processed_equivalent_files > 0 and elapsed_global_now > 0:
                        avg_per_equiv_file = elapsed_global_now / processed_equivalent_files
                        remaining_equiv_files = max(total_files - processed_equivalent_files, 0)
                        global_eta_seconds_now = remaining_equiv_files * avg_per_equiv_file
                    else:
                        global_eta_seconds_now = global_remaining
                    global_eta_lines_str = time_helpers.format_eta(global_eta_seconds_now)

                    if content_length and content_length > 0 and bytes_processed > 0 and elapsed_file > 0:
                        bytes_per_sec = bytes_processed / elapsed_file
                        remaining_bytes = max(content_length - bytes_processed, 0)
                        line_eta_seconds = remaining_bytes / bytes_per_sec if bytes_per_sec > 0 else 0
                        line_eta_str = time_helpers.format_eta(line_eta_seconds)
                        progress_message = (
                            f"{line_count} lines processed | ETA lines: {line_eta_str} | Global ETA: {global_eta_lines_str}"
                        )
                        logger.info(f"    └─ {progress_message}")
                    else:
                        line_eta_str = "N/A"
                        progress_message = (
                            f"{line_count} lines processed | ETA lines: {line_eta_str} | Global ETA: {global_eta_lines_str}"
                        )
                        logger.info(f"    └─ {progress_message}")

                    try:
                        log_to_db(
                            endpoint="normalize_wttj_jobs",
                            level="INFO",
                            message=progress_message,
                            job_id=job_id,
                            dt=dt,
                            output_format=output_format,
                            file=key,
                            files_processed=file_counter,
                            line_count=line_count,
                            line_eta=line_eta_str,
                            global_eta=global_eta_lines_str,
                            status="RUNNING",
                        )
                    except Exception as db_e:
                        logger.warning(f"[normalize_wttj_jobs] log_to_db line progress failed: {db_e}")

                line = line_bytes.decode('utf-8', errors='ignore').strip()
            
                if not line: continue
                
                try:
                    # 1. Parser le JSON
                    record = json.loads(line)
                    #print(f"🔍 Record keys: {list(record.keys())[:5]}")  # DEBUG
                    
                    # 2. Corriger le double encodage
                    record = _fix_double_encoded_dict(record)
                    
                    job_data = get_json_field_from_record(record, "job_data")
                    #print(f"🔍 job_data type: {type(job_data)}, keys: {list(job_data.keys())[:5] if isinstance(job_data, dict) else 'N/A'}")  # DEBUG
                    
                    #initial_data = get_json_field_from_record(record, "initial_data")
                    
                    # Vérifier que job_data existe
                    if not job_data or not isinstance(job_data, dict):
                        print(f"⚠️ job_data invalide ")
                        errors += 1
                        continue
                    
                    #pprint(initial_data)
                    urls_list = job_data.get("urls", [])
                    canonical_url = next((link.get('href', '') for link in urls_list if link.get('kind') == 'canonical'), '')

                    name = clean_html(job_data.get("name", ""))
                    description = clean_html(job_data.get("description", ""))
                    rome_code, rome_label = get_rome_code_from_ml_prediction_via_api(name, description)
                    rome_api_calls += 1
                    profession = _normalize_profession_value(get_field_or_default(record, 'profession'))

                    silver_obj = Silver_Datamodel(
                        id=str(job_data.get("reference", "")),
                        source="WTTJ",
                        url=canonical_url,
                        title=normalize_text(name),
                        description=normalize_text(description),
                        published_at=safe_str_to_datetime(job_data.get("published_at")),
                        updated_at=safe_str_to_datetime(job_data.get("updated_at")),
                        unpublished_at=safe_str_to_datetime(job_data.get("archived_at")),
                        status="archived" if job_data.get("archived_at") else "published",
                        rome_code=rome_code,
                        rome_label=rome_label,
                        title_description="",
                        contract_type=job_data.get("contract_type", ""),
                        worktime="",
                        experience_level=job_data.get("experience_level", ""),
                        experience_description="",
                        naf_code="",
                        job_city="",
                        job_postal_code="",
                        company_name=normalize_text(job_data.get("company_summary", "")),
                        company_city="",
                        company_postal_code="",
                        company_url="",
                        salary_min=safe_str_to_float(job_data.get("salary_min")),
                        salary_max=safe_str_to_float(job_data.get("salary_max")),
                        profile=normalize_text(job_data.get("profile", "")),
                        location_department="",
                        skills=normalize_list_to_strings(job_data.get("skills", [])),
                        salary_periodicity="",
                    )
                    row = silver_obj.to_dict()
                    row["profession"] = profession
                    row["wttj_salary_currency"] = job_data.get("salary_currency")
                    row["wttj_education_level"] = job_data.get("education_level")
                    row["wttj_company_description"] = normalize_text(job_data.get("company_description", ""))
                    row["wttj_contract_duration_min"] = job_data.get("contract_duration_min")
                    row["wttj_contract_duration_max"] = job_data.get("contract_duration_max")
                    row["wttj_remote"] = job_data.get("remote")
                    row["wttj_ats"] = job_data.get("ats")
                    row["wttj_key_missions"] = normalize_list_to_strings(job_data.get("key_missions", []))
                    row["wttj_offices"] = normalize_list_to_strings(job_data.get("offices", []))
                    row["wttj_sectors"] = normalize_list_to_strings(get_field_or_default(record, 'sectors', []))
                    row["wttj_urls"] = urls_list
                    data.append(row)

                    if rome_db_log_every > 0 and (line_count % rome_db_log_every == 0):
                        log_to_db(
                            endpoint="normalize_wttj_jobs_rome_prediction", 
                            level="INFO",
                            message=f"name : {name} - rome_code : {rome_code} | rome_label : {rome_label}",
                            job_id=job_id,
                            dt=dt,
                            output_format=output_format,
                            file=key,
                            files_processed=file_counter,
                            status="RUNNING"
                        )

                except Exception as e:
                    errors += 1
                    print(f"⚠️ Error parsing line in {key}: {e}")
                    # DEBUG: afficher la ligne problématique
                    if errors <= 3:  # Limite à 3 exemples
                        print(f"   Line: {line[:200]}...")

            file_counter += 1
            elapsed = time.time() - start_time
            remaining = (total_files - file_counter) * (elapsed / file_counter) if file_counter > 0 else 0
            eta_str = time_helpers.format_eta(remaining)
            logger.info(f"[normalize_wttj_jobs] Processing {key} | {file_counter}/{total_files} files processed so far | ETA: {eta_str}")
            log_to_db(
                endpoint="normalize_wttj_jobs", 
                level="INFO",
                message=f"Processing {key} | {file_counter}/{total_files} files processed so far | ETA: {eta_str}",
                job_id=job_id,
                dt=dt,
                output_format=output_format,
                file=key,
                files_processed=file_counter,
                ETA=eta_str,
                status="RUNNING"
            )
        except Exception as e:
            errors += 1
            logger.error(f"[normalize_wttj_jobs] Error processing key {key}: {e}")
            try:
                log_to_db(
                    endpoint="normalize_wttj_jobs",
                    level="ERROR",
                    message=f"Error processing key {key}: {e}",
                    job_id=job_id,
                    dt=dt,
                    output_format=output_format,
                    file=key,
                    error=str(e),
                    status="ERROR"
                )
            except Exception as db_e:
                logger.warning(f"[normalize_wttj_jobs] log_to_db error failed: {db_e}")
            continue

    # Save a global dataframe for the entire job (optional, can be heavy if too many records)     
    df = pd.DataFrame(data)
    # storage_silver already in correct directory, just need to write with correct key
    key_base = f"dt={dt}/segment=jobs/all"
    target_key = write_dataframe_with_format(storage_silver, df, key_base, output_format)
    files.append(target_key)
    logger.info(f"[normalize_wttj_jobs] Wrote {output_format}: {target_key} | {len(df)} rows")


    status = "SUCCESS"
    logger.info(
        f"[normalize_wttj_jobs] Done | job_id={job_id} | dt={dt} | output_format={output_format} | "
        f"files={files} | errors={errors} | files counter={file_counter} | "
        f"rome_api_calls={rome_api_calls}"
    )
    try:
        log_to_db(
            endpoint="normalize_wttj_jobs",
            level="INFO",
            message=f"Job done: {job_id}, dt={dt}, output_format={output_format}, files={files}, errors={errors}, files counter={file_counter}",
            job_id=job_id,
            dt=dt,
            output_format=output_format,
            files=files,
            errors=errors,
            status="SUCCESS"
        )
    except Exception as e:
        logger.warning(f"[normalize_wttj_jobs] log_to_db done failed: {e}")
    return NormalizeResult(job_id, status, dt, output_format, files, errors)

def normalize_wttj_jobs_incremental(output_format: str = "parquet") -> NormalizeResult:
    """
    Process only the latest date available in bronze that is not yet in silver.
    This allows to run the normalization incrementally, for example on a daily basis, without reprocessing all historical data.
    """
    # Initialize storage if not provided
    # Source storage : bronze files to normalize
    storage_bronze_merged = get_storage_from_env("bronze", "welcometothejungle")
    #  Destination storage : silver normalized files
    storage_silver_merged = get_storage_from_env("silver", "welcometothejungle")

    logger.info(f"📂 Storage: bronze")
    # List all objects in the storage
    prefix=""
    # Get Immediate prefixes (simulate folders) under bronze
    all_dt_bronze = list(storage_bronze_merged.list_prefixes(prefix))

    dates_bronze = [v.replace('dt=', '').replace('/', '') for v in all_dt_bronze if 'dt=' in v]
    sorted_dates_bronze = sorted(dates_bronze)

    logger.info(f"📂 Storage: silver")
    # List all objects in the storage
    prefix=""
    # Get Immediate prefixes (simulate folders) under silver/merged/
    all_dt_silver = list(storage_silver_merged.list_prefixes(prefix))

    dates_silver = [v.replace('dt=', '').replace('/', '') for v in all_dt_silver if 'dt=' in v  ]
    sorted_dates_silver = sorted(dates_silver)

    dt_to_process = [d for d in sorted_dates_bronze if d not in sorted_dates_silver]    
    if(len(dt_to_process) == 0):
        logger.info("✅ No new date to process. All dates in bronze are already in silver.")
        return NormalizeResult(job_id="none", status="NO_NEW_DATA", dt=None, output_format=output_format, files=[], errors=0)

    ret = NormalizeResult(job_id="incremental", status="RUNNING", dt=str(dt_to_process), output_format=output_format, files=[], errors=0)
    for date in dt_to_process:
        logger.info(f"📂 Date to process: {date}")
        result = normalize_wttj_jobs(date , output_format=output_format)   
        ret.files.extend(result.files) 
        ret.errors += result.errors
        print(f"normalize_wttj_jobs date processed ={date} result: job_id={result.job_id}, status={result.status}, files={result.files}, errors={result.errors}")

    return ret
# ----------------------------
# Main (new / resume / incremental)
# ----------------------------
def main() -> None:
    normalize_wttj_jobs_incremental(output_format="parquet")

if __name__ == "__main__":
    main()