""" 
==============
BRONZE Layer
==============
 Ingest script for Welcome to the Jungle (WTTJ) job offers and company profiles.

This script performs the following steps:

1. Sitemap discovery: It downloads and parses the WTTJ sitemap index to find relevant sitemaps for job listings and company profiles.
2. URL collection: It extracts URLs for job offers and companies from the filtered sitemaps
3. Page fetching: It fetches each URL while respecting rate limits
4. Data processing: Extracts the embedded `window.__INITIAL_DATA__` JSON from the HTML. 
   No transformation is done at this stage, the raw extracted data is stored as-is for maximum flexibility in downstream processing.
5. Storage: It writes the structured data to a storage backend (local or S3) in a chunked manner, along with optional gzipped HTML for debugging.
6. Progress tracking: It maintains a progress meta file with statistics and supports resuming or incremental runs by skipping already processed URLs.

As the site use a client side rendering approach, the embedded `window.__INITIAL_DATA__` contains all the relevant information 
about the job offer or company profile, so we don't need to execute JavaScript or use a headless browser. 

And no html parsing is needed to extract specific fields, which makes the ingestion more efficient and less resource-intensive.

The main function `ingest_welcome_to_the_jungle` can be called from a CLI or scheduled to run periodically (e.g., daily).
It also can be called via a microservice endpoint in FastAPI, allowing for on-demand ingestion or integration with other systems.

"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional, Set, Callable

# ----------------------------
# Import project-specific modules
# ----------------------------
from src.utils.time_helpers import utc_run_id

from src.storage.storage import get_storage_from_env
from src.ingest.tools.rate_limiter import RateLimiter
from src.ingest.tools.welcome_to_the_jungle_common import build_session, collect_urls, load_processed_urls, ingest_segment

# ----------------------------
# Load environment variables
# ----------------------------
from src.config.env import load_project_env
load_project_env()  # safe à rappeler (idempotent)

# ----------------------------
# Logging
# ----------------------------
def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    # Configure thread trace logging if enabled
    trace_enabled = _is_truthy(os.getenv("WTTJ_OPT_THREAD_TRACE_ENABLED", "0"))
    trace_file = os.getenv(
        "WTTJ_OPT_THREAD_TRACE_FILE",
        "logs/ingestion/wttj_thread_trace.log",
    ).strip()

    if trace_enabled and trace_file:
        trace_path = os.path.abspath(trace_file)
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)

        # Reset file handlers so each run starts with a clean trace file
        for handler in list(thread_trace_logger.handlers):
            if isinstance(handler, logging.FileHandler):
                thread_trace_logger.removeHandler(handler)
                handler.close()

        file_handler = logging.FileHandler(trace_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        thread_trace_logger.addHandler(file_handler)

        thread_trace_logger.setLevel(logging.INFO)
        thread_trace_logger.propagate = False
        logger.info("Thread trace file logging enabled: %s", trace_path)


def _is_truthy(value: str) -> bool:
    """Helper to parse environment variable as boolean."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


logger = logging.getLogger("wttj.ingest.bronze")
thread_trace_logger = logging.getLogger("wttj.ingest.bronze.threadtrace")

# ----------------------------
# Sitemap handling
# ----------------------------
# We target the main sitemap index, filter for job listing sitemaps, and extract URLs.
SITEMAP_INDEX = os.getenv("WTTJ_SITEMAP_INDEX", "https://www.welcometothejungle.com/sitemaps/index.xml.gz")
WTTJ_SITEMAP_NS = {"sm": os.getenv("WTTJ_SITEMAP_NS_URL", "http://www.sitemaps.org/schemas/sitemap/0.9")}

# We focus on job listings for now, but company profiles can be added later if needed.
#SITEMAP_ALLOWED_RE = re.compile(r"(company-profiles|job-listings)\.\d+\.xml\.gz$", re.IGNORECASE)
SITEMAP_ALLOWED_RE = re.compile(os.getenv("WTTJ_SITEMAP_ALLOWED_RE", r"(job-listings)\.\d+\.xml\.gz$"), re.IGNORECASE)

# ----------------------------
# Extract window.__INITIAL_DATA__
# ----------------------------
#INITIAL_DATA_RE = re.compile(r'window\.__INITIAL_DATA__\s*=\s*"([\s\S]*?)"\s*//', re.DOTALL)

#INITIAL_DATA_RE = re.compile(
#    r'window\.__INITIAL_DATA__\s*=\s*"(.+?)";',
#    re.DOTALL
#)

# The regex captures the content of window.__INITIAL_DATA__ which is a JSON string, but it may contain escaped quotes and newlines, 
# so we need to unescape it properly before parsing as JSON. 
# The regex looks for the pattern window.__INITIAL_DATA__ = "..." and captures the content inside the quotes, allowing for escaped characters. 
# The re.DOTALL flag allows the dot to match newlines, which is important if the JSON string spans multiple lines.
#INITIAL_DATA_RE = re.compile(
#    r'window\.__INITIAL_DATA__\s*=\s*"((?:\\.|[^"\\])*)"',
#    re.DOTALL
#)

INITIAL_DATA_RE = re.compile(
    os.getenv("WTTJ_INITIAL_DATA_RE", r'window\.__INITIAL_DATA__\s*=\s*"((?:[^"\\]|\\.)*)"\s*;?'),
    re.DOTALL
)

# ----------------------------
# Service function for API
# ----------------------------
def ingest_welcome_to_the_jungle(
    storage=None,
    mode: str = None,
    max_jobs: int = None,
    max_companies: int = None,
    store_html_mode: str = None,
    provided_run_id: str = None,
    workers: int = None,
    part_size: int = None,
    progress_callback: Optional[Callable[[str, int, int, int, int], None]] = None
) -> Dict[str, Any]:
    """
    Service d'ingestion des données Welcome to the Jungle en couche bronze.
    
    Args:
        storage: Storage backend (optionnel, créé depuis env si non fourni)
        mode: Mode d'ingestion (new, resume)
        max_jobs: Limiter le nombre de jobs à traiter (0 = tous)
        max_companies: Limiter le nombre de companies à traiter (0 = tous)
        store_html_mode: Mode de stockage HTML (never, always, on_error)
        provided_run_id: Run ID à utiliser en mode resume
        workers: Nombre de workers concurrents (défaut: 10)
        part_size: Taille des chunks JSONL en nombre de records (défaut: 500)
        progress_callback: Callback appelé avec (segment, current, total, ok, ko)
        
    Returns:
        Dict avec le statut de l'opération et les statistiques détaillées
    """
    try:
        setup_logging()
        
        dt = os.getenv("DT") or datetime.now().date().isoformat()
        
        # Get configuration with defaults
        mode = mode or os.getenv("WTTJ_RUN_MODE", "new").lower().strip()
        provided_run_id = provided_run_id or (os.getenv("WTTJ_RUN_ID") or "").strip()
        store_html_mode = store_html_mode or os.getenv("WTTJ_STORE_HTML", "on_error")
        workers = workers if workers is not None else int(os.getenv("WTTJ_WORKERS", "10"))
        part_size = part_size if part_size is not None else int(os.getenv("WTTJ_PART_SIZE", "500"))
        max_jobs = max_jobs if max_jobs is not None else int(os.getenv("WTTJ_MAX_JOBS", "0"))
        max_companies = max_companies if max_companies is not None else int(os.getenv("WTTJ_MAX_COMPANIES", "0"))
        
        if mode == "resume" and not provided_run_id:
            return {
                "success": False,
                "message": "Mode resume nécessite un run_id",
                "error": "WTTJ_RUN_MODE=resume requires WTTJ_RUN_ID"
            }

        # If provided_run_id is set (e.g. from API task), reuse it for storage and log correlation.
        # Otherwise, generate a fresh run_id.
        run_id = provided_run_id if provided_run_id else utc_run_id()
        
        rps = float(os.getenv("WTTJ_RPS", "2"))
        burst = int(os.getenv("WTTJ_BURST", "4"))
        limiter = RateLimiter(rate=rps, capacity=burst, logger=logger)
        
        session = build_session(workers=workers)
        
        # Initialize storage if not provided
        if storage is None:
            storage = get_storage_from_env("bronze", "welcometothejungle")
        
        logger.info("Début de l'ingestion WTTJ - run_id=%s mode=%s", run_id, mode)
        
        started = time.time()
        
        # Collect URLs from sitemaps
        logger.info("Récupération des URLs depuis les sitemaps...")
        jobs_set, companies_set = collect_urls(session, limiter)
        
        jobs = list(jobs_set)
        companies = list(companies_set)
        
        if max_jobs > 0:
            jobs = jobs[:max_jobs]
            logger.info(f"Limitation à {max_jobs} jobs")
        if max_companies > 0:
            companies = companies[:max_companies]
            logger.info(f"Limitation à {max_companies} companies")
        
        # Skip logic
        skip_jobs: Set[str] = set()
        skip_companies: Set[str] = set()
        
        if mode == "resume":
            jobs_prefix = f"dt={dt}/run_id={run_id}/segment=jobs_raw/"
            comp_prefix = f"dt={dt}/run_id={run_id}/segment=companies_raw/"
            skip_jobs = load_processed_urls(storage, jobs_prefix)
            skip_companies = load_processed_urls(storage, comp_prefix)
            logger.info("Resume mode | already_done jobs=%d | companies=%d", len(skip_jobs), len(skip_companies))
        
        logger.info("Final URL counts | jobs=%d | companies=%d", len(jobs), len(companies))
        
        # Process jobs segment
        jobs_result = ingest_segment(
            dt=dt,
            run_id=run_id,
            segment="jobs",
            urls=jobs,
            storage=storage,
            session=session,
            limiter=limiter,
            store_html_mode=store_html_mode,
            skip_urls=skip_jobs,
            progress_callback=progress_callback,
            part_size=part_size,
            workers=workers 
        )
        
        # Process companies segment
        companies_result = ingest_segment(
            dt=dt,
            run_id=run_id,
            segment="companies",
            urls=companies,
            storage=storage,
            session=session,
            limiter=limiter,
            store_html_mode=store_html_mode,
            skip_urls=skip_companies,
            progress_callback=progress_callback,
            part_size=part_size,
            workers=workers
        )
        
        elapsed = time.time() - started
        total_records = jobs_result["written"] + companies_result["written"]
        throughput = round(total_records / elapsed, 2) if elapsed > 0 else 0
        
        logger.info("Run done | mode=%s | dt=%s | run_id=%s", mode, dt, run_id)
        
        # Emit structured log for monitoring (directement à la source)
        try:
            import json
            from datetime import timezone
            structured_logger = logging.getLogger("structured")
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "ingestion_summary",
                "task_type": "welcome_to_jungle_jobs",
                "run_id": run_id,
                "mode": mode,
                "records_count": total_records,
                "records_written": total_records,
                "duration_sec": round(elapsed, 2),
                "throughput": throughput,
                "jobs_processed": jobs_result["processed"],
                "companies_processed": companies_result["processed"],
                "errors": jobs_result.get("errors", 0) + companies_result.get("errors", 0)
            }
            structured_logger.info(json.dumps(event))
        except Exception as e:
            logger.warning(f"Failed to emit structured log: {e}")
        
        return {
            "success": True,
            "message": f"Ingestion WTTJ terminée avec succès (run_id: {run_id})",
            "run_id": run_id,
            "dt": dt,
            "mode": mode,
            "jobs": jobs_result,
            "companies": companies_result,
            "elapsed_s": elapsed,
            "total_processed": jobs_result["processed"] + companies_result["processed"],
            "total_written": total_records,
            "crawler_version": 1
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'ingestion WTTJ: {e}", exc_info=True)
        return {
            "success": False,
            "message": "Erreur lors de l'ingestion WTTJ",
            "error": str(e)
        }

# ----------------------------
# Main CLI (new / resume / incremental)
# ----------------------------
def main_cli() -> None:
    setup_logging()

    dt = os.getenv("DT") or datetime.now().date().isoformat()

    # Mode:
    # - new: run_id généré, skip_urls = vide
    # - resume: run_id fourni, skip_urls = urls déjà écrites dans CE run
    # - incremental: run_id généré, skip_urls = urls déjà écrites dans un run source
    mode = os.getenv("WTTJ_RUN_MODE", "new").lower().strip()

    provided_run_id = (os.getenv("WTTJ_RUN_ID") or "").strip()
    resume_from_run_id = (os.getenv("WTTJ_RESUME_FROM_RUN_ID") or "").strip()

    if mode == "resume":
        if not provided_run_id:
            raise RuntimeError("WTTJ_RUN_MODE=resume requires WTTJ_RUN_ID")
        run_id = provided_run_id
    else:
        run_id = utc_run_id()

    store_html_mode = os.getenv("WTTJ_STORE_HTML", "on_error")

    rps = float(os.getenv("WTTJ_RPS", "2"))
    burst = int(os.getenv("WTTJ_BURST", "4"))
    limiter = RateLimiter(rate=rps, capacity=burst, logger =logger)

    # Read workers and part_size from environment or use defaults
    workers = int(os.getenv("WTTJ_WORKERS", "10"))
    part_size = int(os.getenv("WTTJ_PART_SIZE", "500"))

    session = build_session(workers=workers)
    
    # Storage (local / S3) — structure unifiée bronze/welcometothejungle
    storage = get_storage_from_env("bronze", "welcometothejungle")

    logger.info("Run start | mode=%s | dt=%s | run_id=%s", mode, dt, run_id)

    # Collect URLs from sitemaps
    jobs_set, companies_set = collect_urls(session, limiter)

    max_jobs = int(os.getenv("WTTJ_MAX_JOBS", "0"))
    max_companies = int(os.getenv("WTTJ_MAX_COMPANIES", "0"))

    # We convert the sets to lists and apply any configured limits on the number of URLs to process for jobs and companies.
    jobs = list(jobs_set)
    companies = list(companies_set)

    # for testing or debugging purposes by setting WTTJ_MAX_JOBS or WTTJ_MAX_COMPANIES to a positive integer.
    if max_jobs > 0:
        jobs = jobs[:max_jobs]
    if max_companies > 0:
        companies = companies[:max_companies]

    # Skip logic
    skip_jobs: Set[str] = set()
    skip_companies: Set[str] = set()

    # Depending on the run mode, we determine which URLs to skip based on previously processed records in storage.
    if mode == "resume":
        jobs_prefix = f"dt={dt}/run_id={run_id}/segment=jobs_raw/"
        comp_prefix = f"dt={dt}/run_id={run_id}/segment=companies_raw/"
        skip_jobs = load_processed_urls(storage, jobs_prefix)
        skip_companies = load_processed_urls(storage, comp_prefix)
        logger.info("Resume mode | already_done jobs=%d | companies=%d", len(skip_jobs), len(skip_companies))

    elif mode == "incremental":
        if not resume_from_run_id:
            raise RuntimeError("WTTJ_RUN_MODE=incremental requires WTTJ_RESUME_FROM_RUN_ID")
        jobs_prefix = f"dt={dt}/run_id={resume_from_run_id}/segment=jobs_raw/"
        comp_prefix = f"dt={dt}/run_id={resume_from_run_id}/segment=companies_raw/"
        skip_jobs = load_processed_urls(storage, jobs_prefix)
        skip_companies = load_processed_urls(storage, comp_prefix)
        logger.info(
            "Incremental mode | base_run_id=%s | skip jobs=%d | companies=%d",
            resume_from_run_id, len(skip_jobs), len(skip_companies)
        )

    logger.info("Final URL counts | jobs=%d | companies=%d", len(jobs), len(companies))

    # We then call the ingest_segment function for both jobs and companies.
    ingest_segment(
        dt=dt,
        run_id=run_id,
        segment="jobs",
        urls=jobs,
        storage=storage,
        session=session,
        limiter=limiter,
        store_html_mode=store_html_mode,
        workers=workers,
        part_size=part_size,
        skip_urls=skip_jobs,
    )

    ingest_segment(
        dt=dt,
        run_id=run_id,
        segment="companies",
        urls=companies,
        storage=storage,
        session=session,
        limiter=limiter,
        store_html_mode=store_html_mode,
        workers=workers,
        part_size=part_size,
        skip_urls=skip_companies,
    )

    logger.info("Run done | mode=%s | dt=%s | run_id=%s", mode, dt, run_id)


if __name__ == "__main__":
    main_cli()