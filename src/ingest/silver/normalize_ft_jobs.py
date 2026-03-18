"""Normalize France Travail bronze offers into canonical silver records.

Responsibilities of this file:
- Source-specific normalization for France Travail (bronze -> silver).
- HTML/text cleanup, list normalization, and basic type harmonization (na management).
- Output of canonical `Silver_Datamodel`-compatible rows under `dt=...`.

Out of scope for this file:
- Cross-source deduplication (FT vs WTTJ).
- Cross-source harmonization rules for merged datasets.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict
from typing import List

import pandas as pd
from tqdm import tqdm

from src.config.env import load_project_env
from src.ingest.data_models.silver_datamodel_class import NormalizeResult, Silver_Datamodel
from src.storage.storage import get_storage_from_env
from src.utils.log_to_db import log_to_db
from src.utils.normalization_helpers import safe_str_to_datetime, safe_str_to_float, write_dataframe_with_format
from src.utils.storage_tools import get_last_dt_from_storage
from src.utils.text_processing import normalize_list_to_strings, normalize_text

load_project_env()  # safe to call multiple times

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_seconds(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _extract_run_prefix(key: str) -> str:
    parts = key.split("/")
    if len(parts) >= 3 and parts[0] == "offers" and parts[1].startswith("dt=") and parts[2].startswith("run_id="):
        return "/".join(parts[:3]) + "/"
    last_sep = key.rfind("/")
    return (key[: last_sep + 1] if last_sep >= 0 else key)


def normalize_ft_jobs(dt: str, output_format: str = "parquet") -> NormalizeResult:
    """ 
    Normalize FT jobs for a given date (dt) and save to silver.

    Operations:
    - Load JSONL records from bronze for the specified date prefix.
    - For each record, extract and normalize fields into the Silver_Datamodel structure.
    - Handle missing/NA values gracefully with safe parsing functions (like `safe_str_to_datetime` and `safe_str_to_float`).
    - Save the resulting DataFrame to silver in the specified output format under a structured key.

    Args:
    - dt: Date to process (format YYYYMMDD). If "latest" or empty, processes the latest date available in bronze.
    - output_format: Format to save the normalized data (parquet or jsonl).

    """
    job_id = f"ft-normalize-{dt}-{uuid.uuid4().hex[:8]}"
    files: List[str] = []
    errors = 0

    storage_bronze = get_storage_from_env("bronze", "france_travail")
    storage_silver = get_storage_from_env("silver", "france_travail")

    if dt is None or dt == "" or dt == "latest":
        latest_key = get_last_dt_from_storage(storage_bronze, "offers/")
        dt = latest_key if latest_key else ""

    # Get all JSONL keys for the specified date prefix
    prefix = f"offers/dt={dt}/"
    logger.info(f"[normalize_ft_jobs] Load keys for prefix: {prefix}")
    keys = [k for k in storage_bronze.list_keys(prefix) if k.endswith(".jsonl")]

    logger.info(
        f"[normalize_ft_jobs] Start | dt={dt} | output_format={output_format} | "
        f"job_id={job_id} | files={len(keys)}"
    )

    try:
        log_to_db(
            endpoint="normalize_ft_jobs",
            level="INFO",
            message=f"Start job: {job_id}, dt={dt}, output_format={output_format}, files={len(keys)}",
            job_id=job_id,
            dt=dt,
            output_format=output_format,
            files=len(keys),
            status="RUNNING",
        )
    except Exception as exc:
        logger.warning(f"[normalize_ft_jobs] log_to_db start failed: {exc}")

    data = []
    total_files = len(keys)
    file_counter = 0
    global_start_time = time.time()

    #For tqdm progress bars, group keys by run_id prefix to show progress per run_id batch
    keys_by_run = defaultdict(list)
    for key in keys:
        keys_by_run[_extract_run_prefix(key)].append(key)

    for run_prefix in sorted(keys_by_run.keys()):
        run_keys = keys_by_run[run_prefix]
        logger.info(f"[normalize_ft_jobs] Processing {run_prefix}")
        progress_bar = tqdm(run_keys, desc=f"Processing {run_prefix}", unit="file", mininterval=15, leave=False)

        for key in progress_bar:
            try:
                for record in storage_bronze.read_jsonl(key):
                    try:
                        origine_offre = record.get("origineOffre", {}) if isinstance(record, dict) else {}
                        if not isinstance(origine_offre, dict):
                            origine_offre = {}
                        url = origine_offre.get("urlOrigine", "")

                        lieu_travail = record.get("lieuTravail", {}) if isinstance(record, dict) else {}
                        if not isinstance(lieu_travail, dict):
                            lieu_travail = {}
                        job_city = lieu_travail.get("libelle", "")
                        job_postal_code = lieu_travail.get("codePostal", "")
                        company_city = lieu_travail.get("commune", "")
                        location_department = job_postal_code[:2] if job_postal_code else ""

                        entreprise = record.get("entreprise", {}) if isinstance(record, dict) else {}
                        if not isinstance(entreprise, dict):
                            entreprise = {}
                        company_name = entreprise.get("nom", "")

                        salaire = record.get("salaire", {}) if isinstance(record, dict) else {}
                        if not isinstance(salaire, dict):
                            salaire = {}
                        salary_text = salaire.get("libelle", "")

                        silver_obj = Silver_Datamodel(
                            id=str(record.get("id", "")),
                            source="FT",
                            url=url,
                            title=normalize_text(record.get("intitule", "")),
                            description=normalize_text(record.get("description", "")),
                            published_at=safe_str_to_datetime(record.get("dateCreation", "")),
                            updated_at=safe_str_to_datetime(record.get("dateActualisation", "")),
                            unpublished_at=None,
                            status="published",
                            rome_code=record.get("romeCode", ""),
                            rome_label=record.get("romeLibelle", ""),
                            title_description=normalize_text(record.get("appellationlibelle", "")),
                            contract_type=record.get("typeContratLibelle", ""),
                            worktime="" if pd.isna(record.get("dureeTravailLibelleConverti")) else record.get("dureeTravailLibelleConverti", ""),
                            experience_level=record.get("experienceExige", ""),
                            experience_description=record.get("experienceLibelle", ""),
                            naf_code=record.get("codeNAF", ""),
                            job_city=job_city,
                            job_postal_code=job_postal_code,
                            company_name=company_name,
                            company_city=company_city,
                            company_postal_code=job_postal_code,
                            company_url=record.get("entreprise_url", ""),
                            salary_min=safe_str_to_float(salary_text),
                            salary_max=safe_str_to_float(salary_text),
                            profile=normalize_text(record.get("profile", "")),
                            location_department=location_department,
                            skills=normalize_list_to_strings(record.get("skills", [])),
                            salary_periodicity=str(salary_text) if salary_text is not None else "",
                        )
                        data.append(silver_obj.to_dict())
                    except Exception as exc:
                        errors += 1
                        logger.warning(f"[normalize_ft_jobs] Error parsing record in {run_prefix}: {exc}")

                file_counter += 1
                elapsed = time.time() - global_start_time
                estimated_total = (elapsed / file_counter) * total_files if file_counter > 0 else 0
                remaining = max(estimated_total - elapsed, 0)
                elapsed_str = _format_seconds(elapsed)
                estimated_total_str = _format_seconds(estimated_total)
                eta_remaining_str = _format_seconds(remaining)
                progress_bar.set_postfix_str(
                    f"{file_counter}/{total_files} elapsed={elapsed_str} eta={eta_remaining_str}"
                )

                try:
                    log_to_db(
                        endpoint="normalize_ft_jobs",
                        level="INFO",
                        message=(
                            f"Processing {run_prefix} | {file_counter}/{total_files} files processed | "
                            f"elapsed={elapsed_str} | estimated_total={estimated_total_str} | eta_remaining={eta_remaining_str}"
                        ),
                        job_id=job_id,
                        dt=dt,
                        output_format=output_format,
                        run_prefix=run_prefix,
                        files_processed=file_counter,
                        elapsed=elapsed_str,
                        estimated_total=estimated_total_str,
                        ETA=eta_remaining_str,
                        status="RUNNING",
                    )
                except Exception as exc:
                    logger.warning(f"[normalize_ft_jobs] log_to_db progress(done file) failed: {exc}")
            except Exception as exc:
                errors += 1
                logger.error(f"[normalize_ft_jobs] Error processing {run_prefix}: {exc}")

    df = pd.DataFrame(data)
    key_base = f"dt={dt}/all"
    target_key = write_dataframe_with_format(storage_silver, df, key_base, output_format)

    files.append(target_key)

    logger.info(
        f"[normalize_ft_jobs] Done | job_id={job_id} | dt={dt} | output_format={output_format} | "
        f"rows={len(df)} | files={files} | errors={errors}"
    )

    try:
        log_to_db(
            endpoint="normalize_ft_jobs",
            level="INFO",
            message=f"Job done: {job_id}, dt={dt}, output_format={output_format}, files={files}, errors={errors}",
            job_id=job_id,
            dt=dt,
            output_format=output_format,
            files=files,
            errors=errors,
            status="SUCCESS",
        )
    except Exception as exc:
        logger.warning(f"[normalize_ft_jobs] log_to_db done failed: {exc}")

    return NormalizeResult(job_id, "SUCCESS", dt, output_format, files, errors, rows=len(df))


def normalize_ft_jobs_incremental(output_format: str = "parquet") -> NormalizeResult:
    """Normalize only FT dates present in bronze and absent in silver."""
    storage_bronze = get_storage_from_env("bronze", "france_travail")
    storage_silver = get_storage_from_env("silver", "france_travail")

    bronze_dt_prefixes = list(storage_bronze.list_prefixes("offers/"))
    bronze_dates = sorted(
        d.replace("dt=", "").replace("/", "")
        for d in bronze_dt_prefixes
        if "dt=" in d
    )

    silver_dt_prefixes = list(storage_silver.list_prefixes(""))
    silver_dates = sorted(
        d.replace("dt=", "").replace("/", "")
        for d in silver_dt_prefixes
        if "dt=" in d
    )

    dates_to_process = [d for d in bronze_dates if d not in silver_dates]
    if not dates_to_process:
        return NormalizeResult(
            job_id="none",
            status="NO_NEW_DATA",
            dt=None,
            output_format=output_format,
            files=[],
            errors=0,
        )

    aggregate = NormalizeResult(
        job_id="incremental",
        status="RUNNING",
        dt=str(dates_to_process),
        output_format=output_format,
        files=[],
        errors=0,
    )

    for date in dates_to_process:
        result = normalize_ft_jobs(date, output_format=output_format)
        aggregate.files.extend(result.files)
        aggregate.errors += result.errors

    aggregate.status = "SUCCESS"
    return aggregate


def main() -> None:
    output_format = os.getenv("FT_NORMALIZE_OUTPUT_FORMAT", "parquet")
    normalize_ft_jobs_incremental(output_format=output_format)


if __name__ == "__main__":
    main()
