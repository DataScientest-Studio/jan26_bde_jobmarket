""" 
==============
Normalize WTTJ Jobs
==============

Parse bronze files from Welcome to the Jungle, clean and normalize data, enrich with ROME code prediction, 
and write to silver layer.

Conserve the same directory structure (dt=.../segment=jobs) but with cleaned data and new format (parquet/jsonl/csv).

Arguments:
- dt: date of the data to process (format YYYY-MM-DD) in bronze layer
- output_format: parquet (default), jsonl, or csv        

Method is expose in a CLI entry point (main) and can be called by API in order to process data programmatically.
For example with airflow or any scheduler.
"""
from __future__ import annotations
import os
import uuid
import time
import pandas as pd
import json
from typing import List
import src.ingest.tools.time_helpers as time_helpers

from src.utils.wttj_utils import get_field_or_default, find_field_in_json, _fix_double_encoded_dict, get_json_field_from_record
from src.utils.text_processing import clean_html
from src.utils.storage_tools import get_last_dt_from_storage
from src.utils.rome import get_rome_code_from_ml_prediction

from src.config.env import require_env, get_project_root, load_project_env
load_project_env()  # safe à rappeler (idempotent)

from src.storage.storage import get_storage_from_env
from src.utils.log_to_db import log_to_db

import src.ingest.tools.time_helpers as time_helpers
from src.ingest.data_models.welcome_to_the_jungle_class import WTTJ

# Pour forcer l’affichage en console pour tous les loggers,
# à ajouter avant toute création de logger :
import logging
logging.basicConfig(level=logging.INFO)
# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger(__name__)
structured_logger = logging.getLogger("structured")

class NormalizeWTTJResult:
    def __init__(self, job_id, status, dt, output_format, files, errors):
        self.job_id = job_id
        self.status = status
        self.dt = dt
        self.format = output_format
        self.files = files
        self.errors = errors


def normalize_wttj_jobs(dt: str, output_format: str = "parquet") -> NormalizeWTTJResult:
    job_id = f"wttj-normalize-{dt}-{uuid.uuid4().hex[:8]}"
    status = "RUNNING"
    files: List[str] = []
    errors = 0
    storage_bronze = get_storage_from_env("bronze", "welcometothejungle")
    storage_silver = get_storage_from_env("silver", "welcometothejungle")
    # Optimized: list only runid folders, then segment=jobs_raw keys per runid
    if( dt is None or dt == "" or dt == "latest"):
        dt = get_last_dt_from_storage(storage_bronze, "")

    runid_prefix = f"dt={dt}/"
    runid_folders = storage_bronze.list_prefixes(runid_prefix)
    jobs_raw_keys = []
    start_time = time.time()
    for runid_folder in runid_folders:
        segment_prefix = runid_prefix + runid_folder + "segment=jobs_raw/"
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

    file_counter = 0
    for key in jobs_raw_keys:
        data = []
        try:
            obj = storage_bronze.get_object_jsonl(key=key)
            # Iterlines to preserve memory load
            for line_bytes in obj["Body"].iter_lines():
                #print(line_bytes[:3000])
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
                    rome_code, rome_label = get_rome_code_from_ml_prediction(name, description)
                    profession = get_field_or_default(record, 'profession')

                    wttj =WTTJ(
                        reference=job_data.get("reference"),
                        name=name,
                        description=description,
                        profile=clean_html(job_data.get("profile")),
                        salary_min=job_data.get("salary_min"),
                        salary_max=job_data.get("salary_max"),
                        salary_currency=job_data.get("salary_currency"),
                        education_level=job_data.get("education_level"),
                        company_summary=job_data.get("company_summary"),
                        company_description=job_data.get("company_description"),
                        updated_at=job_data.get("updated_at"),
                        published_at=job_data.get("published_at"),
                        archived_at=job_data.get("archived_at"),
                        contract_duration_min=job_data.get("contract_duration_min"),
                        remote=job_data.get("remote"),
                        ats=job_data.get("ats"),
                        contract_duration_max=job_data.get("contract_duration_max"),
                        experience_level=job_data.get("experience_level"),
                        contract_type=job_data.get("contract_type"),
                        urls=urls_list,
                        canonical_url=canonical_url,
                        skills=job_data.get("skills", [""]),
                        key_missions=job_data.get("key_missions", [""]),
                        offices=job_data.get("offices", [""]),
                        sectors=get_field_or_default(record, 'sectors', []),
                        profession=profession
                        )
                    row = vars(wttj)
                    row["rome_code"] = rome_code
                    row["rome_label"] = rome_label
                    data.append(row)

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
    silver_key = f"dt={dt}/segment=jobs"
    if output_format == "parquet":
        storage_silver.write_parquet(silver_key.replace(".jsonl", ".parquet"), df)
        files.append(silver_key.replace(".jsonl", ".parquet"))
        logger.info(f"[normalize_wttj_jobs] Wrote parquet: {silver_key.replace('.jsonl', '.parquet')} | {len(df)} rows")
    elif output_format == "jsonl":
        storage_silver.write_jsonl(silver_key.replace(".jsonl", ".jsonl"), df.to_dict(orient="records"))
        files.append(silver_key.replace(".jsonl", ".jsonl"))
        logger.info(f"[normalize_wttj_jobs] Wrote jsonl: {silver_key.replace('.jsonl', '.jsonl')} | {len(df)} rows")
    elif output_format == "csv":
        storage_silver.write_bytes(silver_key.replace(".jsonl", ".csv"), df.to_csv(index=False).encode("utf-8"), content_type="text/csv")
        files.append(silver_key.replace(".jsonl", ".csv"))
        logger.info(f"[normalize_wttj_jobs] Wrote csv: {silver_key.replace('.jsonl', '.csv')} | {len(df)} rows")


    status = "SUCCESS"
    logger.info(f"[normalize_wttj_jobs] Done | job_id={job_id} | dt={dt} | output_format={output_format} | files={files} | errors={errors} | files counter={file_counter}")
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
    return NormalizeWTTJResult(job_id, status, dt, output_format, files, errors)

# ----------------------------
# Main (new / resume / incremental)
# ----------------------------
def main() -> None:
    #dt = "2026-02-28"
    dt=""
    result = normalize_wttj_jobs(dt, output_format="parquet")
    print(f"normalize_wttj_jobs result: job_id={result.job_id}, status={result.status}, files={result.files}, errors={result.errors}")


if __name__ == "__main__":
    main()