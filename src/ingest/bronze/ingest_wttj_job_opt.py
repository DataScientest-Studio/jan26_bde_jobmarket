"""
Welcome to the Jungle - Ingestion optimisée du crawl de WTTJ
==========================================================
Version est intégrée au système storage du projet (S3/local).

Usage programmatique:
    from src.ingest.bronze.ingest_wttj_jobs_opt import ingest_welcome_to_the_jungle_opt
    
    result = ingest_welcome_to_the_jungle_opt(
        mode="new",
        max_jobs=1000,
        workers=8
    )

Usage CLI:
    python -m src.ingest.bronze.ingest_wttj_jobs_opt

"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Callable

try:
    from curl_cffi import requests as cffi_requests
    USE_CFFI = True
except ImportError:
    import requests as cffi_requests
    USE_CFFI = False

# ─────────────────────────────────────────────
#  Import project-specific modules
# ─────────────────────────────────────────────
from src.utils.time_helpers import utc_run_id, utc_now_iso
from src.storage.storage import get_storage_from_env
from src.ingest.data_models.bronze_datamodel_class import wtt_bronze_datamodels

# ─────────────────────────────────────────────
#  Load environment variables
# ─────────────────────────────────────────────
from src.config.env import load_project_env
load_project_env()  # safe à rappeler (idempotent)

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

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

        # Reset file handlers so each run starts with a clean trace file.
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

logger = logging.getLogger("wttj.ingest.bronze.jobs_opt")
thread_trace_logger = logging.getLogger("wttj.ingest.bronze.jobs_opt.threadtrace")


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

# ─────────────────────────────────────────────
#  Constantes API
# ─────────────────────────────────────────────
API_BASE = os.getenv("WTTJ_API_BASE", "https://api.welcometothejungle.com/api/v1")

HEADERS_API = {
    "Accept": "application/json",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.welcometothejungle.com/",
    "Origin": "https://www.welcometothejungle.com",
    "User-Agent": os.getenv(
        "WTTJ_UA",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

HEADERS_HTML = {
    "User-Agent": HEADERS_API["User-Agent"],
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9",
}


# ─────────────────────────────────────────────
#  Extraction slugs depuis URL
# Based on: https://www.welcometothejungle.com/fr/companies/{org}/jobs/{job}
# ─────────────────────────────────────────────
def extract_slugs(url: str) -> Optional[tuple[str, str]]:
    """Extrait (org_slug, job_slug) depuis une URL WTTJ."""
    match = re.search(r'/companies/([^/]+)/jobs/([^/?#]+)', url)
    if match:
        return match.group(1), match.group(2)
    logger.warning(f"Impossible d'extraire les slugs depuis : {url}")
    return None


# ─────────────────────────────────────────────
#  Fetch API REST
# ─────────────────────────────────────────────
def fetch_job_api(org_slug: str, job_slug: str, timeout: int = 15) -> tuple[Optional[dict], Optional[int]]:
    """Appelle l'API REST WTTJ et retourne (dict, status_code)."""
    url = f"{API_BASE}/organizations/{org_slug}/jobs/{job_slug}"
    logger.debug(f"  API call : {url}")
    try:
        if USE_CFFI:
            resp = cffi_requests.get(
                url, headers=HEADERS_API, impersonate="chrome120", timeout=timeout
            )
        else:
            resp = cffi_requests.get(url, headers=HEADERS_API, timeout=timeout)
        
        # Gestion spécifique des codes d'erreur
        if resp.status_code == 429:
            logger.warning(f"  ⚠️  [429 Rate Limited] ({org_slug}/{job_slug})")
            return None, 429
        elif resp.status_code == 403:
            logger.warning(f"  🚫 [403 Forbidden] ({org_slug}/{job_slug})")
            return None, 403
        elif resp.status_code == 404:
            logger.debug(f"  ℹ️  [404 Not Found] ({org_slug}/{job_slug})")
            return None, 404
        elif resp.status_code >= 500:
            logger.warning(f"  ⚠️  [HTTP {resp.status_code}] ({org_slug}/{job_slug})")
            return None, resp.status_code
        
        resp.raise_for_status()
        return resp.json(), 200
    except Exception as e:
        logger.warning(f"  ⚠️  API error ({org_slug}/{job_slug}) : {type(e).__name__}")
        return None, None


# ─────────────────────────────────────────────
#  Fetch HTML (fallback)
# ─────────────────────────────────────────────
def fetch_html(url: str, timeout: int = 20) -> tuple[Optional[str], Optional[int]]:
    """Fetch HTML brut pour le fallback JSON-LD. Retourne (html, status_code)"""
    try:
        if USE_CFFI:
            resp = cffi_requests.get(
                url, headers=HEADERS_HTML, impersonate="chrome120", timeout=timeout
            )
        else:
            resp = cffi_requests.get(url, headers=HEADERS_HTML, timeout=timeout)
        
        if resp.status_code == 429:
            logger.warning(f"  ⚠️  [429 Rate Limited] {url}")
            return None, 429
        elif resp.status_code == 403:
            logger.warning(f"  🚫 [403 Forbidden] {url}")
            return None, 403
        elif resp.status_code == 404:
            logger.debug(f"  ℹ️  [404 Not Found] {url}")
            return None, 404
        elif resp.status_code >= 500:
            logger.warning(f"  ⚠️  [HTTP {resp.status_code}] {url}")
            return None, resp.status_code
        
        resp.raise_for_status()
        return resp.text, 200
    except Exception as e:
        logger.error(f"  ⚠️  HTML error ({url}) : {type(e).__name__}")
        return None, None


# ─────────────────────────────────────────────
#  Extraction JSON-LD JobPosting (fallback)
# ─────────────────────────────────────────────
def extract_jsonld_job(html: str) -> Optional[dict]:
    """Extrait le JSON-LD de type JobPosting depuis le HTML statique."""
    pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    for raw in re.findall(pattern, html, re.DOTALL):
        try:
            data = json.loads(raw.strip())
            if data.get("@type") in ("JobPosting", "jobPosting"):
                return data
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def safe_get(d, *keys, default=None):
    """Safely navigate nested dictionaries."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
        if d is None:
            return default
    return d


def join_list(lst, key="name", sep=" | "):
    """Join list of dicts by a key."""
    if not lst or not isinstance(lst, list):
        return None
    items = [
        str(item.get(key, ""))
        for item in lst
        if isinstance(item, dict) and item.get(key)
    ]
    return sep.join(items) if items else None


# ─────────────────────────────────────────────
#  Compute job key (compatible avec ingest_wttj_jobs.py)
# ─────────────────────────────────────────────
def compute_job_key(job_data: Optional[Dict[str, Any]], url: str) -> str:
    """Compute a unique key for a job offer (compatible with main pipeline)."""
    if isinstance(job_data, dict):
        for k in ("reference", "wttj_reference", "slug"):
            v = job_data.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────
#  Build record — API (compatible bronze datamodel)
# ─────────────────────────────────────────────
def build_record_from_api(data: dict, source_url: str) -> tuple[dict, dict]:
    """
    Construit le record depuis la réponse de l'API WTTJ.
    Retourne (initial_data, job_data) compatibles avec wtt_bronze_datamodels.
    """
    job = data.get("job", data)
    if not isinstance(job, dict):
        job = data

    # initial_data: données brutes de l'API (équivalent de window.__INITIAL_DATA__)
    initial_data = {
        "source_method": "wttj_api_rest",
        "raw_api_response": data
    }
    
    # job_data: données structurées extraites (format compatible avec pick_job_data)
    job_data = job

    return initial_data, job_data


# ─────────────────────────────────────────────
#  Build record — JSON-LD fallback (compatible bronze datamodel)
# ─────────────────────────────────────────────
def build_record_from_jsonld(jsonld: dict, source_url: str) -> tuple[dict, dict]:
    """
    Construit un record depuis le JSON-LD JobPosting (fallback HTML).
    Retourne (initial_data, job_data) compatibles avec wtt_bronze_datamodels.
    """
    # initial_data: données brutes JSON-LD
    initial_data = {
        "source_method": "wttj_jsonld_fallback",
        "raw_jsonld": jsonld
    }
    
    # job_data: transformation du JSON-LD en format rapproché de l'API
    hiring_org = jsonld.get("hiringOrganization", {}) or {}
    location = jsonld.get("jobLocation", {}) or {}
    if isinstance(location, list):
        location = location[0] if location else {}
    address = location.get("address", {}) or {}
    if isinstance(address, list):
        address = address[0] if address else {}
    salary_data = jsonld.get("baseSalary", {}) or {}
    salary_val = salary_data.get("value", {}) or {}

    emp_type_map = {
        "FULL_TIME": "full_time", "PART_TIME": "part_time",
        "CONTRACTOR": "freelance", "INTERN": "internship",
        "TEMPORARY": "temp_work", "OTHER": None,
    }
    raw_emp = jsonld.get("employmentType", "")
    contract = emp_type_map.get(raw_emp, raw_emp.lower() if raw_emp else None)

    # Nettoyage de la description HTML
    description_raw = jsonld.get("description", "") or ""
    description_clean = re.sub(r"<[^>]+>", " ", description_raw)
    description_clean = re.sub(r"\s+", " ", description_clean).strip() or None

    job_data = {
        "name": jsonld.get("title"),
        "description": description_clean,
        "profile": jsonld.get("qualifications"),
        "contract_type": contract,
        "published_at": jsonld.get("datePosted"),
        "expires_at": jsonld.get("validThrough"),
        "status": "published",
        "apply_url": jsonld.get("url"),
        "industry": jsonld.get("industry"),
        "salary": {
            "minimum": salary_val.get("minValue"),
            "maximum": salary_val.get("maxValue"),
            "currency": salary_data.get("currency"),
            "period": "yearly"
        },
        "offices": [{
            "address": address.get("streetAddress"),
            "zip_code": address.get("postalCode"),
            "city": address.get("addressLocality"),
            "state": address.get("addressRegion"),
            "country_code": address.get("addressCountry")
        }],
        "organization": {
            "name": hiring_org.get("name"),
            "website_url": hiring_org.get("sameAs"),
            "logo_url": hiring_org.get("logo")
        }
    }

    return initial_data, job_data


# ─────────────────────────────────────────────
#  Pipeline d'extraction principale
# ─────────────────────────────────────────────
def scrape_url(url: str) -> tuple[Optional[tuple[dict, dict, int]], Optional[int]]:
    """
    Pipeline d'extraction pour une URL WTTJ :
      1. Extraction des slugs depuis l'URL
      2. Appel API REST → record complet
      3. Fallback : fetch HTML + JSON-LD
    
    Retourne ((initial_data, job_data, status_code), error_code) où error_code est None si succès,
    ou le code HTTP d'erreur (429, 403, etc.) sinon.
    """

    # ── Étape 1 : API REST ───────────────────────────────────────────────
    #slugs = extract_slugs(url)
    #if slugs :
    #    org_slug, job_slug = slugs
    #    data, api_error = fetch_job_api(org_slug, job_slug)
    #    if data:
    #        initial_data, job_data = build_record_from_api(data, source_url=url)
    #        return (initial_data, job_data, 200), None
    #    if api_error in (429, 403):  # Ban ou rate limit
    #        return None, api_error
    #    logger.debug(f"  API sans résultat, passage au fallback HTML pour : {url}")

    # ── Étape 2 : Fallback HTML + JSON-LD ───────────────────────────────
    html, html_error = fetch_html(url)
    if html:
        #jsonld = extract_jsonld_job(html)
        #if jsonld:
        #    logger.debug("  Source: JSON-LD fallback")
        #    initial_data, job_data = build_record_from_jsonld(jsonld, source_url=url)
        #    return (initial_data, job_data, 200), None

        # Get Row Data
        from  src.ingest.tools.welcome_to_the_jungle_common import extract_initial_data_from_html, pick_job_data 
        initial_data = extract_initial_data_from_html(html)
        # We only pick the job_data if we successfully extracted the initial_data and if we're in the "jobs" segment.
        if initial_data :
            job_data = pick_job_data(initial_data)

        return (initial_data, job_data, 200), None
    
    if html_error in (429, 403):  # Ban ou rate limit
        return None, html_error

    logger.warning(f"  Aucune donnée extraite pour : {url}")
    return None, None


# ─────────────────────────────────────────────
#  Service d'ingestion
# ─────────────────────────────────────────────
def ingest_welcome_to_the_jungle_opt(
    storage=None,
    mode: str = None,
    max_urls: int = None,
    workers: int = None,
    part_size: int = None,
    delay: float = None,
    provided_run_id: str = None,
    progress_callback: Optional[Callable[[str, int, int, int, int], None]] = None,
    force_download_urls: bool = False
) -> Dict[str, Any]:
    """
    Service d'ingestion des données Welcome to the Jungle via API REST optimisée.
    
    Args:
        storage: Storage backend (optionnel, créé depuis env si non fourni)
        mode: Mode d'ingestion (new, resume, incremental)           
        max_urls: Limiter le nombre d'URLs à traiter (0 = tous)
        workers: Nombre de workers concurrents (défaut: 8)
        part_size: Taille des chunks JSONL en nombre de records (défaut: 500)
        delay: Délai entre requêtes par thread en secondes (défaut: 0.5)
        provided_run_id: Run ID à utiliser
        progress_callback: Callback appelé avec (segment, current, total, ok, ko)
        
    Returns:
        Dict avec le statut de l'opération et les statistiques détaillées
    """
    try:
        setup_logging()
        
        dt = os.getenv("DT") or datetime.now().date().isoformat()
        
        # Get configuration with defaults
        mode = mode or os.getenv("WTTJ_OPT_RUN_MODE", "new").lower().strip()
        provided_run_id = provided_run_id or (os.getenv("WTTJ_OPT_RUN_ID") or "").strip()
        workers = workers if workers is not None else int(os.getenv("WTTJ_WORKERS", "8"))
        part_size = part_size if part_size is not None else int(os.getenv("WTTJ_PART_SIZE", "500"))
        max_urls = max_urls if max_urls is not None else int(os.getenv("WTTJ_MAX_JOBS", "0"))
        delay = delay if delay is not None else float(os.getenv("WTTJ_THREAD_DELAY", "1"))
        
        # Generate or use provided run_id
        run_id = provided_run_id if provided_run_id else utc_run_id()
        
        # Initialize storage if not provided
        if storage is None:
            storage = get_storage_from_env("bronze", "welcometothejungle")
        
        logger.info("Début de l'ingestion WTTJ OPT - run_id=%s mode=%s", run_id, mode)
        
        try:
            if force_download_urls:
                from src.ingest.bronze.ingest_wttj_collect_urls import collect_sitemap_urls
                logger.info(f" force_download_urls={force_download_urls}-  Collecting URLs from sitemap...")
                ret = collect_sitemap_urls(max_results=max_urls if max_urls > 0 else 0)
                if ret.get("success", False) is  True:
                    logger.info(f"Loaded {ret.get('urls_count', 0)} URLs download")

            # Get urls from storage 
            storage_urls_key = "sitemap/urls.txt"
            urls_text = storage.read_bytes(storage_urls_key).decode("utf-8")
            urls = [line.strip() for line in urls_text.split("\n") if line.strip()]
            logger.info(f"Loaded {len(urls)} URLs from storage ({storage_urls_key})")

        except Exception as e:
            logger.debug(f"Could not read from storage {storage_urls_key}: {e}")
            urls=   None
            # Get URLs from sitemap if not loaded from storage
        
        if not urls:
            return {
                "success": False,
                "message": "No URLs to process",
                "error": "No URLs provided or collected"
            }
        
        if max_urls > 0:
            urls = urls[:max_urls]
            logger.info(f"Limitation à {max_urls} URLs")
        
        logger.info(f"Processing {len(urls)} URLs with {workers} workers | delay: {delay:.1f}s per request")
        
        started = time.time()
        
        # Storage setup (use same prefix pattern as main pipeline)
        raw_prefix = f"dt={dt}/run_id={run_id}/segment=jobs_raw/"
        
        # Process URLs
        result = process_urls_segment(
            urls=urls,
            storage=storage,
            raw_prefix=raw_prefix,
            dt=dt,
            run_id=run_id,
            workers=workers,
            part_size=part_size,
            delay=delay,
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - started
        elapsed_int = int(elapsed)
        elapsed_h = elapsed_int // 3600
        elapsed_m = (elapsed_int % 3600) // 60
        elapsed_s = elapsed_int % 60
        elapsed_hhmmss = f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"

        logger.info(
            "Run done | mode=%s | dt=%s | run_id=%s | urls=%d | elapsed=%s | errors=%d",
            mode,
            dt,
            run_id,
            len(urls),
            elapsed_hhmmss,
            result.get("ko", 0),
        )
        
        return {
            "success": True,
            "message": f"Ingestion WTTJ OPT terminée avec succès (run_id: {run_id})",
            "run_id": run_id,
            "dt": dt,
            "mode": mode,
            "jobs_opt": result,
            "elapsed_s": elapsed,
            "total_processed": result["processed"],
            "total_written": result["written"]
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'ingestion WTTJ OPT: {e}", exc_info=True)
        return {
            "success": False,
            "message": "Erreur lors de l'ingestion WTTJ OPT",
            "error": str(e)
        }


def process_urls_segment(
    *,
    urls: List[str],
    storage,
    raw_prefix: str,
    dt: str,
    run_id: str,
    workers: int,
    part_size: int,
    delay: float,
    progress_callback: Optional[Callable[[str, int, int, int, int], None]] = None
) -> Dict[str, Any]:
    """Process a list of URLs and write results to storage in chunks."""
    
    total_urls = len(urls)
    meta_key = f"{raw_prefix}run.json"
    
    # Get next part number
    part_no = next_part_no(storage, raw_prefix)
    
    ok = 0
    ko = 0
    processed = 0
    total_written = 0
    buffer: List[dict] = []
    
    buffer_lock = threading.Lock()
    stats_lock = threading.Lock()
    
    start_time = time.time()
    last_log_time = start_time
    last_log_processed = 0
    trace_enabled = _is_truthy(os.getenv("WTTJ_OPT_THREAD_TRACE_ENABLED", "0"))
    
    logger.info(f"Start processing | urls={total_urls} | workers={workers} | part_size={part_size} | next_part={part_no}")
    
    def flush_buffer() -> None:
        """Flush the current buffer of records to storage as a new part file."""
        nonlocal part_no, total_written, buffer
        if not buffer:
            return

        part_key = f"{raw_prefix}part-{part_no:06d}.jsonl"
        written = storage.write_jsonl(part_key, buffer)
        total_written += written
        logger.info("Wrote %s | records=%d | total_written=%d", part_key, written, total_written)

        part_no += 1
        buffer = []

        write_progress_meta(storage, meta_key, {
            "source": "welcometothejungle",
            "dt": dt,
            "run_id": run_id,
            "segment": "jobs",
            "input_urls_total": total_urls,
            "processed": processed,
            "ok": ok,
            "ko": ko,
            "parts_written": part_no - 1,
            "written": total_written,
            "updated_at": utc_now_iso(),
        })
    
    def process_one(url: str) -> None:
        """Process a single URL."""
        nonlocal ok, ko, processed, buffer, last_log_time, last_log_processed

        thread_name = threading.current_thread().name
        fetch_started_at = time.time()
        if trace_enabled:
            thread_trace_logger.info(
                "event=FETCH_START thread=%s url=%s",
                thread_name,
                url,
            )

        result, error_code = scrape_url(url)
        fetch_elapsed_ms = int((time.time() - fetch_started_at) * 1000)
        trace_status = 200 if result is not None else (error_code if error_code is not None else "NA")
        if trace_enabled:
            thread_trace_logger.info(
                "event=FETCH_DONE thread=%s status=%s elapsed_ms=%d url=%s",
                thread_name,
                trace_status,
                fetch_elapsed_ms,
                url,
            )

        fetched_at = utc_now_iso()
        
        # Build record OUTSIDE the lock to avoid blocking other threads
        if result is None:
            # Create error record using wtt_bronze_datamodels
            if error_code == 429:
                error_msg = "rate_limited_429"
                status_code = 429
            elif error_code == 403:
                error_msg = "forbidden_403"
                status_code = 403
            elif error_code:
                error_msg = f"http_error_{error_code}"
                status_code = error_code
            else:
                error_msg = "scrape_failed"
                status_code = None
            
            record = asdict(wtt_bronze_datamodels(
                source="welcometothejungle",
                segment="jobs",
                url=url,
                fetched_at=fetched_at,
                status_code=status_code,
                ok=False,
                error=error_msg,
                key=hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
                initial_data={},
                job_data={},
                parser_version=2,
            ))
            is_success = False
        else:
            # Unpack result
            initial_data, job_data, status_code = result
            
            # Compute key
            key_id = compute_job_key(job_data, url)
            
            # Create record using wtt_bronze_datamodels
            record = asdict(wtt_bronze_datamodels(
                source="welcometothejungle",
                segment="jobs",
                url=url,
                fetched_at=fetched_at,
                status_code=status_code,
                ok=True,
                error="",
                key=key_id,
                initial_data=initial_data if initial_data else {},
                job_data=job_data if job_data else {},
                parser_version=2,
            ))
            is_success = True
        
        # MINIMAL critical section: just update counters
        with stats_lock:
            processed += 1
            if is_success:
                ok += 1
            else:
                ko += 1
            
            # Add to buffer
            buffer.append(record)
            needs_flush = len(buffer) >= part_size
            
            # Track progress
            should_log = processed % 100 == 0 or processed == total_urls
        
        # Flush OUTSIDE the stats_lock if needed
        if needs_flush:
            with buffer_lock:
                if len(buffer) >= part_size:
                    flush_buffer()
        
        # Progress callback
        if progress_callback:
            progress_callback("jobs", processed, total_urls, ok, ko)
        
        # Log progress OUTSIDE the lock
        if should_log:
            with stats_lock:
                current_time = time.time()
                
                # Calculate instantaneous rate (since last log)
                time_since_last_log = current_time - last_log_time
                urls_since_last_log = processed - last_log_processed
                
                if time_since_last_log > 0 and urls_since_last_log > 0:
                    rate = urls_since_last_log / time_since_last_log
                else:
                    # Fallback to average rate if first log
                    elapsed = current_time - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                
                # Calculate ETA
                remaining = total_urls - processed
                eta_seconds = remaining / rate if rate > 0 else 0
                
                if eta_seconds > 0:
                    eta_hours = int(eta_seconds // 3600)
                    eta_mins = int((eta_seconds % 3600) // 60)
                    eta_secs = int(eta_seconds % 60)
                    eta_str = f"ETA: {eta_hours:02d}:{eta_mins:02d}:{eta_secs:02d}"
                else:
                    eta_str = "ETA: --:--:--"
                
                logger.info(f"  [{processed:5d}/{total_urls}] | OK: {ok} | KO: {ko} | Rate: {rate:.2f} urls/s | {eta_str}")
                
                # Update last log trackers
                last_log_time = current_time
                last_log_processed = processed
        
        # Apply per-thread delay to respect rate limits (same as scraper2.py)
        if delay > 0:
            sleep_seconds = random.uniform(delay, delay * 1.3)
            if trace_enabled:
                thread_trace_logger.info(
                    "event=SLEEP_START thread=%s sleep_s=%.3f url=%s",
                    thread_name,
                    sleep_seconds,
                    url,
                )

            sleep_started_at = time.time()
            time.sleep(sleep_seconds)

            if trace_enabled:
                thread_trace_logger.info(
                    "event=SLEEP_DONE thread=%s slept_ms=%d url=%s",
                    thread_name,
                    int((time.time() - sleep_started_at) * 1000),
                    url,
                )
    
    # Process URLs with thread pool
    logger.info(f"Starting ThreadPoolExecutor with {workers} workers...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, url): url for url in urls}
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                url = futures[future]
                logger.error(f"Exception processing {url}: {e}")
                with stats_lock:
                    ko += 1
                    processed += 1
    
    # Flush remaining buffer
    flush_buffer()
    
    # Write final progress meta
    write_progress_meta(storage, meta_key, {
        "source": "welcometothejungle",
        "dt": dt,
        "run_id": run_id,
        "segment": "jobs",
        "input_urls_total": total_urls,
        "processed": processed,
        "ok": ok,
        "ko": ko,
        "parts_written": part_no - 1,
        "written": total_written,
        "updated_at": utc_now_iso(),
        "completed": True
    })
    
    return {
        "segment": "jobs",
        "processed": processed,
        "written": total_written,
        "ok": ok,
        "ko": ko,
        "parts_written": part_no - 1,
        "crawler_version": 2,
    }


def next_part_no(storage, prefix: str) -> int:
    """Determine the next part number based on existing files."""
    try:
        keys = storage.list_keys(prefix)
        part_numbers = []
        for key in keys:
            match = re.search(r'part-(\d+)\.jsonl$', key)
            if match:
                part_numbers.append(int(match.group(1)))
        return max(part_numbers, default=-1) + 1
    except Exception as e:
        logger.warning(f"Could not determine next part number: {e}")
        return 0


def write_progress_meta(storage, key: str, data: dict) -> None:
    """Write progress metadata to storage."""
    try:
        storage.write_json(key, data)
    except Exception as e:
        logger.error(f"Failed to write progress meta: {e}")


# ─────────────────────────────────────────────
#  Main CLI
# ─────────────────────────────────────────────
def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingestion WTTJ optimisée via API REST"
    )
    parser.add_argument("--max-urls", type=int, default=10, help="Nombre maximum d'URLs à traiter")
    parser.add_argument("--workers", type=int, default=4, help="Nombre de workers concurrents")
    parser.add_argument("--delay", type=float, default=2, help="Délai entre requêtes (secondes)")
    parser.add_argument("--verbose", action="store_true", help="Active les logs DEBUG")
    parser.add_argument("--force-download-urls", default=False, help="Force le téléchargement des URLs depuis le sitemap à chaque run")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    result = ingest_welcome_to_the_jungle_opt(
        max_urls=args.max_urls,
        workers=args.workers,
        delay=args.delay,
        force_download_urls=args.force_download_urls  # Forcer le téléchargement des URLs depuis le sitemap à chaque run
    )
       
    
    if not result["success"]:
        logger.error("Ingestion failed")
        exit(1)



if __name__ == "__main__":
    main()
