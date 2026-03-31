import os
import logging
import requests
import gzip
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import json
from io import BytesIO
from tqdm import tqdm  
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable, Callable
import re
import time
from dataclasses import asdict

from src.utils.time_helpers import utc_now_iso, format_eta
from src.utils.log_to_db import log_to_db
import src.ingest.tools.rate_limiter as RateLimiter

# Data models
from src.ingest.data_models.bronze_datamodel_class import wtt_bronze_datamodels
from concurrent.futures import ThreadPoolExecutor 
import threading

# ----------------------------
# Load environment variables
# ----------------------------
from src.config.env import load_project_env

logger = logging.getLogger("wttj.ingest.bronze")
thread_trace_logger = logging.getLogger("wttj.ingest.bronze.threadtrace")
load_project_env()  # safe à rappeler (idempotent)


def _is_truthy(value: str) -> bool:
    """Helper to parse environment variable as boolean."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def setup_thread_trace_logging() -> bool:
    """
    Configure thread trace logging if enabled via environment variables.
    Returns True if trace logging is enabled and configured successfully.
    """
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
        return True
    
    return False

SITEMAP_INDEX = os.getenv("WTTJ_SITEMAP_INDEX", "https://www.welcometothejungle.com/sitemaps/index.xml.gz")
WTTJ_SITEMAP_NS= {"sm": os.getenv("WTTJ_SITEMAP_NS_URL", "http://www.sitemaps.org/schemas/sitemap/0.9")}    
SITEMAP_ALLOWED_RE = re.compile(os.getenv("WTTJ_SITEMAP_ALLOWED_RE", r"(job-listings)\.\d+\.xml\.gz$"), re.IGNORECASE)
INITIAL_DATA_RE = re.compile(
    os.getenv("WTTJ_INITIAL_DATA_RE", r'window\.__INITIAL_DATA__\s*=\s*"((?:[^"\\]|\\.)*)"\s*;?'),
    re.DOTALL
)

# ----------------------------
# HTTP Session with Retry
# ----------------------------
def build_session(workers: int = 10) -> requests.Session:
    """ 
    Build a requests session with retry logic and custom headers.
    Pool size is calibrated to the number of workers to avoid holding
    more connections open than needed, which wastes memory and file descriptors.
    """

    session = requests.Session()

    # Pool size calibrated to workers.
    # +2 for margin (sitemap downloads, occasional burst)
    pool_size = workers + 2
    
    # Configure retries with exponential backoff for transient errors
    retry = Retry(
        total=int(os.getenv("WTTJ_RETRIES", "5")),
        backoff_factor=float(os.getenv("WTTJ_BACKOFF", "0.6")),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    # Use a connection pool with a reasonable size for concurrent requests
    adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=pool_size,
            pool_maxsize=pool_size,
        )    
    # Mount the adapter for both HTTP and HTTPS
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Set a realistic User-Agent to avoid being blocked by WTTJ. 
    # TODO : Rotate User-Agents if needed, or use a library like fake-useragent for more variability.
    session.headers.update(        {
            "User-Agent": os.getenv(
                "WTTJ_UA",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120 Safari/537.36",
            )
        }
    )

    return session


def download_gz_xml(session: requests.Session, url: str, limiter: RateLimiter) -> bytes:
    """ Download a gzipped XML file and return its content as bytes. """
    limiter.acquire()
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def parse_gz_sitemap_locations(payload: bytes, tag: str = "loc") -> List[str]:
    """ Parse a gzipped XML sitemap and return a list of URLs. """
    xml_bytes: bytes
    # Detect if the payload is gzipped by checking the magic number (first two bytes)
    is_gzip = len(payload) >= 2 and payload[0] == 0x1F and payload[1] == 0x8B

    if is_gzip:
        try:
            with gzip.GzipFile(fileobj=BytesIO(payload)) as f:
                xml_bytes = f.read()
        except gzip.BadGzipFile:
            xml_bytes = payload
    else:
        xml_bytes = payload

    # Quick check to see if the content looks like XML or if it's an HTML error page (e.g., blocked or redirected)
    head = xml_bytes[:200].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        snippet = xml_bytes[:300].decode("utf-8", errors="replace")
        logger.error("Sitemap payload looks like HTML (blocked/redirect?). Snippet: %s", snippet)
        return []

    root = ET.fromstring(xml_bytes)
    locs = []
    # Extract the text content of all <loc> elements (or the specified tag) in the sitemap XML
    # .//sm means  = find all descendants with the sm namespace and the given tag
    for el in root.findall(f".//sm:{tag}", WTTJ_SITEMAP_NS):
        if el.text:
            locs.append(el.text.strip())
    return locs


def list_target_sitemaps(session: requests.Session, limiter: RateLimiter) -> List[str]:
    """ Download the sitemap index, parse it, and return a list of filtered sitemap URLs. """
    logger.info("Downloading sitemap index: %s", SITEMAP_INDEX)
    gz = download_gz_xml(session, SITEMAP_INDEX, limiter)
    sitemap_urls = parse_gz_sitemap_locations(gz, tag="loc")

    filtered = [u for u in sitemap_urls if SITEMAP_ALLOWED_RE.search(u)]
    logger.info("Sitemaps in index=%d | filtered=%d", len(sitemap_urls), len(filtered))
    return filtered


def add_url(url: str, target: Set[str], kind: str) -> None:
    """ Add a URL to the target set if it's not already present, and log the action."""
    if url in target:
        logger.info("Duplicate %s ignored: %s | total=%d", kind, url, len(target))
        return
    target.add(url)
    logger.info("Added %s: %s | total=%d", kind, url, len(target))


def collect_urls(session: requests.Session, limiter: RateLimiter) -> Tuple[Set[str], Set[str]]:
    sitemaps = list_target_sitemaps(session, limiter)
    jobs, companies = set(), set()
    
    pbar_sitemaps = tqdm(sitemaps, desc="📄 Sitemaps")
    for sm_url in pbar_sitemaps:
        gz = download_gz_xml(session, sm_url, limiter)
        urls = sorted(parse_gz_sitemap_locations(gz, tag="loc"))
        
        # COMPTEUR URLS (pas logs)
        job_urls = [u for u in urls if "job-listings" in sm_url]
        new_jobs = add_urls_batch(job_urls, jobs)
        
        pbar_sitemaps.set_postfix({
            "Jobs": len(jobs), 
            "Sitemap": f"+{new_jobs}"
        })
    
    print(f"🎉 {len(jobs)} jobs | {len(companies)} companies")
    return jobs, companies


def add_urls_batch(urls, url_set):
    """Ajoute batch, retourne NB nouveaux"""
    before = len(url_set)
    url_set.update(urls)
    return len(url_set) - before


def extract_initial_data_from_html(html: str) -> Optional[Dict[str, Any]]:
    """ Extract and parse the JSON data from window.__INITIAL_DATA__ in the HTML. Returns a dict or None if not found or parsing fails. """
    m = INITIAL_DATA_RE.search(html)
    if not m:
        return None

    # 1. Extract the raw JSON string (still escaped as it is in the HTML)
    raw_string = m.group(1)

    try:
        # 2. Décode Unicode JS (\uXXXX → UTF-8)
        json_text = bytes(raw_string, 'utf-8').decode('unicode_escape')
        
        # 3. Parse JSON principal
        data = json.loads(json_text)
        
        # 4. Dé-sérialise les champs internes (arrays/objects stringifiés)
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    data[key] = parsed
                    print(f"🔄 {key}: string → {type(parsed).__name__}")
                except json.JSONDecodeError:
                    pass  # Garde string si pas JSON
        return data
    except Exception as e:
        print(f"Error parsing initial data: {e}")
        return None


def pick_job_data(initial_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """ From the extracted initial data, pick the relevant job data section. The structure may vary, so we look for a query with queryKey starting with "job" and return its state.data if it's a dict. """
    queries = initial_data.get("queries") or []
    for q in queries:
        qk = q.get("queryKey") or []
        data = (q.get("state") or {}).get("data")
        if isinstance(qk, list) and qk and qk[0] == "job" and isinstance(data, dict):
            return data
    return None


def compute_job_key(job_data: Optional[Dict[str, Any]], url: str) -> str:
    """ Compute a unique key for a job offer. If the job_data contains a reference, wttj_reference, or slug, we use that (after stripping). 
    Otherwise, we fallback to a hash of the URL. 
    This ensures that we have a stable and human-readable key when possible, and a consistent fallback when not. 
    This key is also used to map the job offer to its HTML gzipped file if we choose to store it.
    """
    if isinstance(job_data, dict):
        for k in ("reference", "wttj_reference", "slug"):
            v = job_data.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def compute_company_key(url: str) -> str:
    """ Compute a unique key for a company profile. We try to extract the company slug from the URL using a regex."""
    m = re.search(r"/companies/([^/]+)", url)
    if m:
        return m.group(1)
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

# ----------------------------
# Storage helpers (resume / incremental)
# ----------------------------
# We write the raw extracted data in chunks (e.g., 500 records per file) to the storage backend.
PART_RE = re.compile(r"part-(\d+)\.jsonl$")

def next_part_no(storage, raw_prefix: str) -> int:
    """
    Determine the next part number to write based on existing part-*.jsonl files under the given raw_prefix.
    raw_prefix ex:
      bronze/dt=YYYY-MM-DD/run_id=RUN/segment=jobs_raw/
    """
    try:
        keys = list(storage.list_keys(raw_prefix))
    except Exception:
        return 1

    max_no = 0
    for k in keys:
        m = PART_RE.search(k)
        if m:
            max_no = max(max_no, int(m.group(1)))
    return max_no + 1


def iter_existing_records(storage, raw_prefix: str) -> Iterable[Dict[str, Any]]:
    """
    Iterate over existing records in the raw_prefix by reading all part-*.jsonl files.
    """
    try:
        keys = list(storage.list_keys(raw_prefix))
    except Exception:
        return []

    part_keys = sorted([k for k in keys if PART_RE.search(k)])
    for pk in part_keys:
        for rec in storage.read_jsonl(pk):
            if isinstance(rec, dict):
                yield rec


def load_processed_urls(storage, raw_prefix: str) -> Set[str]:
    """ Load the set of already processed URLs from existing records in the raw_prefix. 
    Used for resuming or incremental runs to skip already processed URLs. """
    processed: Set[str] = set()
    for rec in iter_existing_records(storage, raw_prefix):
        url = rec.get("url")
        if isinstance(url, str) and url:
            processed.add(url)
    return processed


def write_progress_meta(storage, meta_key: str, payload: Dict[str, Any]) -> None:
    """ Write a progress meta file (e.g., run.json) with the given payload. 
    File contains statistics about the ingestion progress and can be used for monitoring and resuming. """
    try:
        storage.write_json(meta_key, payload)
    except Exception as e:
        logger.warning("Failed to write progress meta: %s | error=%s", meta_key, e)


# ----------------------------
# Writing (chunked + safe flush)
# ----------------------------
def should_store_html(mode: str, res_ok: bool, initial_data_ok: bool) -> bool:
    """ Determine whether to store the HTML content based on the configured mode and the result of the fetch and data extraction."""
    m = (mode or "on_error").lower().strip()
    if m == "never":
        return False
    if m == "always":
        return True
    return (not res_ok) or (not initial_data_ok)


class FetchResult:
    """Class to represent the result of fetching a page, including the URL, status, HTML content, and any error message."""
    def __init__(self, url: str, ok: bool, status_code: Optional[int], html: Optional[str], error: Optional[str]):
        self.url = url
        self.ok = ok
        self.status_code = status_code
        self.html = html
        self.error = error


def fetch_page(session: requests.Session, limiter, url: str, trace_enabled: bool = False) -> FetchResult:
    """Fetch avec pool de proxies"""
    thread_name = threading.current_thread().name
    
    if trace_enabled:
        thread_trace_logger.info(
            "event=FETCH_START thread=%s url=%s",
            thread_name,
            url,
        )
    
    fetch_started_at = time.time()
    limiter.acquire()
    
    try:
        r = session.get(url, timeout=30)
        fetch_elapsed_ms = int((time.time() - fetch_started_at) * 1000)
        
        if trace_enabled:
            thread_trace_logger.info(
                "event=FETCH_DONE thread=%s status=%s elapsed_ms=%d url=%s",
                thread_name,
                r.status_code,
                fetch_elapsed_ms,
                url,
            )
        
        if r.status_code >= 400:
            return FetchResult(url=url, ok=False, status_code=r.status_code, html=r.text, 
                             error=f"HTTP {r.status_code}")
        
        return FetchResult(url=url, ok=True, status_code=r.status_code, html=r.text, error=None)
    
    except Exception as e:
        fetch_elapsed_ms = int((time.time() - fetch_started_at) * 1000)
        
        if trace_enabled:
            thread_trace_logger.info(
                "event=FETCH_DONE thread=%s status=ERROR elapsed_ms=%d url=%s error=%s",
                thread_name,
                fetch_elapsed_ms,
                url,
                str(e),
            )
        
        logger.error(f"[FETCH ERROR] {url}: {e}")
        return FetchResult(url=url, ok=False, status_code=None, html=None, error=str(e))


def ingest_segment(
    *,
    dt: str,
    run_id: str,
    segment: str,
    urls: List[str],
    storage,
    session: requests.Session,
    limiter: RateLimiter,
    store_html_mode: str,
    workers: int = 10,
    part_size: int = 500,
    skip_urls: Optional[Set[str]] = None,
    progress_callback: Optional[Callable[[str, int, int, int, int], None]] = None,
) -> Dict[str, Any]:
    """
    The function processes the URLs in a concurrent manner using a thread pool, while respecting the rate limiter. 
    It extracts the relevant data from each page, writes structured records to storage in chunks, and optionally stores the raw HTML for debugging. 
    Progress is tracked and written to a meta file for monitoring and resuming purposes.

    Parameters:
    - dt: The date of the run (e.g., "2024-06-01")
    - run_id: The unique identifier for this run (e.g., "20240601T120000Z")
    - segment: The segment name (e.g., "jobs" or "companies")
    - urls: The list of URLs to process for this segment
    - storage: The storage backend instance to write data to
    - session: The HTTP session to use for fetching pages
    - limiter: The rate limiter instance to control request rates
    - store_html_mode: Configuration for when to store HTML content ("never", "always", "on_error")
    - workers: Number of concurrent worker threads (default: 10)
    - part_size: Number of records to write per JSONL chunk file (default: 500)
    - skip_urls: An optional set of URLs to skip (e.g., already processed in a previous run)

    Returns:
    - Dict with segment statistics (processed, written, ok, ko, elapsed_s)
    """

    # We use a thread pool to fetch pages concurrently while respecting the rate limiter.

    # We determine the list of URLs to process by excluding any URLs in the skip_urls set.
    skip_urls = skip_urls or set()
    
    # This allows us to resume or run incrementally without reprocessing URLs we've already handled in previous runs, 
    # which is important for efficiency and avoiding duplicate work.
    urls_todo = [u for u in urls if u not in skip_urls]
    # Url to process (not skipped) count
    total_urls = len(urls_todo)

    # Progress log frequency based on total URLs.
    # We want frequent feedback, so log every 100 URLs or 1% of total (whichever is smaller).
    # This ensures visibility even for large ingestion jobs.
    progress_log_every = max(1, min(100, total_urls // 100 if total_urls > 0 else 1))
    
    # We define a prefix for the raw data files in storage based on the date, run ID, and segment.
    raw_prefix = f"dt={dt}/run_id={run_id}/segment={segment}_raw/"

    # We also define a key for the progress meta file (e.g., run.json) that will contain statistics about the ingestion progress.
    meta_key = f"{raw_prefix}run.json"

    # We determine the next part number to write based on existing files in the raw_prefix. 
    # This ensures that we don't overwrite existing data and can continue writing new parts sequentially.
    part_no = next_part_no(storage, raw_prefix)

    ok = 0
    ko = 0
    processed = 0
    total_written = 0
    buffer: List[wtt_bronze_datamodels] = []

    start_time = time.time()
    last_checkpoint_time = start_time
    
    logger.info(
        f"Start segment={segment} | urls_total={len(urls)} | skip={len(skip_urls)} | "
        f"todo={len(urls_todo)} | workers={workers} | part_size={part_size} | next_part={part_no}"
    )    

    # Standardized task_id prefix for this segment operation (easier filtering in DB/Grafana)
    segment_task_prefix = f"{run_id}:segment={segment}"
    segment_task_id = f"{segment_task_prefix}:event=start"
    
    log_to_db(
        endpoint='welcome_to_the_jungle',
        level='INFO',
        message=f"📥 Début ingestion segment {segment}: {len(urls_todo)} URLs à traiter ({len(skip_urls)} déjà traitées)",
        task_id=segment_task_id,
        duration_sec=0,
        records_count=len(urls_todo),
        extra_metadata={
            'segment': segment,
            'urls_total': len(urls),
            'skip': len(skip_urls),
            'workers': workers,
            'part_size': part_size,
            'run_id': run_id
        }
    )


    def flush_buffer() -> None:
        """ 
        Nested function to flush the current buffer of records to storage as a new part file, and update the progress meta.
        Do a garbage collection after flushing to give back memory to the OS, which is important in a long-running ingestion process to avoid memory bloat.
        """
        nonlocal part_no, total_written, buffer
        if not buffer:
            return

        part_key = f"{raw_prefix}part-{part_no:06d}.jsonl"
        written = storage.write_jsonl(part_key, buffer)
        total_written += written
        logger.info("Wrote %s | records=%d | total_written=%d", part_key, written, total_written)

        part_no += 1
        buffer = []

        # Garbage Collector Python : give back memory to the OS after each flush
        # Without this, the allocator keeps the memory fragmented even after buffer = []
        import gc
        gc.collect() 

        write_progress_meta(storage, meta_key, {
            "source": "welcometothejungle",
            "dt": dt,
            "run_id": run_id,
            "segment": segment,
            "input_urls_total": len(urls),
            "input_urls_skipped": len(skip_urls),
            "input_urls_todo": total_urls,
            "processed": processed,
            "ok": ok,
            "ko": ko,
            "parts_written": part_no - 1,
            "written": total_written,
            "updated_at": utc_now_iso(),
        })

    # If there are no URLs to process (e.g., all URLs were skipped), we write the progress meta with zero processed and exit early.
    if total_urls == 0:
        logger.info("Nothing to do for segment=%s (all urls skipped).", segment)
        write_progress_meta(storage, meta_key, {
            "source": "welcometothejungle",
            "dt": dt,
            "run_id": run_id,
            "segment": segment,
            "input_urls_total": len(urls),
            "input_urls_skipped": len(skip_urls),
            "input_urls_todo": 0,
            "processed": 0,
            "ok": 0,
            "ko": 0,
            "parts_written": part_no - 1,
            "written": 0,
            "updated_at": utc_now_iso(),
        })
        segment_task_id_empty = f"{segment_task_prefix}:event=empty"
        log_to_db(
            endpoint='welcome_to_the_jungle',
            level='INFO',
            message=f"⏭️ Segment {segment}: aucune URL à traiter (toutes déjà importées)",
            task_id=segment_task_id_empty,
            duration_sec=0.0,
            records_count=0,
            error_count=0,
            extra_metadata={
                'segment': segment,
                'reason': 'all_urls_skipped',
                'skipped': len(skip_urls),
                'run_id': run_id
            }
        )
        return {
            "segment": segment,
            "processed": 0,
            "written": 0,
            "ok": 0,
            "ko": 0,
            "elapsed_s": 0.0
        }
 
    # Check if thread trace logging is enabled
    trace_enabled = _is_truthy(os.getenv("WTTJ_OPT_THREAD_TRACE_ENABLED", "0"))
    if trace_enabled:
        setup_thread_trace_logging()

     # ── Batch streaming de Futures ────────────────────────────────────────────
     # Instead of submitting all URLs at once (which could lead to 39k futures in memory simultaneously),
     # We keep a sliding window of SUBMIT_BATCH active futures at any given time.
     # It limit upper memory usage to ~SUMBIT_BATCH pages HTML in memory, 
     # instead of potentially several GB if all pages were retained simultaneously.
    SUBMIT_BATCH = 200  # Max futures  simultaneous active 

    from concurrent.futures import wait, FIRST_COMPLETED

    # We use a ThreadPoolExecutor to fetch pages concurrently. 
    # For each URL, we submit a fetch_page task to the pool, which will return a FetchResult when done.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        
        urls_iter = iter(urls_todo)
        # Dictionnaire des futures actifs : future → url
        # On le maintient petit (≤ SUBMIT_BATCH) pour limiter la mémoire
        active_futures: dict = {}

        def fill_queue():
            """
            Remplit la queue de futures jusqu'à SUBMIT_BATCH actifs.
            Appelé au démarrage puis après chaque batch de completions
            pour maintenir un flux constant de requêtes sans saturer la mémoire.
            """
            while len(active_futures) < SUBMIT_BATCH:
                url = next(urls_iter, None)
                if url is None:
                    break  # Plus d'URLs à soumettre
                fut = pool.submit(fetch_page, session, limiter, url)
                active_futures[fut] = url

        logger.info(
            f"⏳ Streaming {len(urls_todo)} URLs | batch_size={SUBMIT_BATCH} | workers={workers}"
        )

       # First queue set 
        fill_queue()

        # Main loop : we wait for at least one future to complete,
        # process all completed futures, then resubmit to maintain SUBMIT_BATCH active.
        while active_futures:
            # Wait for at least one future to complete before processing
            # FIRST_COMPLETED avoids busy-waiting and lets threads work
            done, _ = wait(active_futures, return_when=FIRST_COMPLETED)

            for fut in done:
                res = fut.result()
                #  Immediately release the Future from the active queue.
                # Without this del, the dict would grow indefinitely even with wait().
                del active_futures[fut]

                fetched_at = utc_now_iso()
                processed += 1

                # Log the first processing to confirm workers are active
                if processed == 1:
                    logger.info(
                        f"🚀 First URL processed. Workers active. "
                        f"Rate limiter at {limiter.rate} req/s"
                    )

               # Data extract
                initial_data = None
                job_data = None

                if res.html:
                    initial_data = extract_initial_data_from_html(res.html)
                    # We only pick the job_data if we successfully extracted the initial_data
                    # and if we're in the "jobs" segment.
                    if initial_data and segment == "jobs":
                        job_data = pick_job_data(initial_data)

                if segment == "jobs":
                    key_id = compute_job_key(job_data, res.url)
                else:
                    key_id = compute_company_key(res.url)

                # Convert the record to a datamodel instance,
                # which ensures consistent structure and types, and then to a dict for storage (via asdict).
                # We use dataclasses for the bronze datamodels, so asdict is appropriate here (.dict is for pydantic).
                record = asdict(wtt_bronze_datamodels(
                    source="welcometothejungle",
                    segment=segment,
                    url=res.url,
                    fetched_at=fetched_at,
                    status_code=res.status_code,
                    ok=res.ok,
                    error=res.error,
                    key=key_id,
                    initial_data=initial_data if initial_data else {},
                    job_data=job_data if job_data else {},
                    parser_version=1,
                ))

                # ── HTML gz optionnel ────────────────────────────────────────
                # Store HTML before releasing res, as we need res.html for writing
                if res.html and should_store_html(store_html_mode, res.ok, initial_data is not None):
                    html_key = (
                        f"dt={dt}/run_id={run_id}/"
                        f"segment={segment}_html/key={key_id}/page.html.gz"
                    )
                    try:
                        storage.write_gzip_text(html_key, res.html)
                    except Exception as e:
                        logger.exception("Failed writing html.gz | %s | %s", html_key, e)

                # ← Libère explicitement le FetchResult (contient res.html, potentiellement
                # 100KB-1MB par page). Sans ce del, Python attend le prochain GC cycle
                # pour libérer, ce qui peut représenter plusieurs GB en cours de traitement.
                del res

                # ── Compteurs ────────────────────────────────────────────────
                if record.get("ok"):
                    ok += 1
                else:
                    ko += 1

                buffer.append(record)

                # Flush batch when buffer reaches the part size,
                # to avoid keeping too much data in memory and to write incrementally to storage.
                if len(buffer) >= part_size:
                    flush_buffer()

                # Callback de progression (utilisé par l'API pour mise à jour temps réel)
                if progress_callback:
                    progress_callback(segment, processed, total_urls, ok, ko)

                # ── Progress log (% + ETA) ───────────────────────────────────
                if processed % progress_log_every == 0 or processed == total_urls:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    remaining = total_urls - processed
                    eta_seconds = remaining / rate if rate > 0 else 0.0
                    pct = (processed / total_urls) * 100 if total_urls > 0 else 0.0

                    # Calculate batch duration (time since last checkpoint)
                    current_time = time.time()
                    batch_duration = current_time - last_checkpoint_time

                    logger.info(
                        "Progress segment=%s | done=%d/%d (%.2f%%) | ok=%d ko=%d | "
                        "rate=%.2f req/s | active_futures=%d | elapsed=%s | ETA=%s",
                        segment,
                        processed,
                        total_urls,
                        pct,
                        ok,
                        ko,
                        rate,
                        len(active_futures),
                        format_eta(elapsed),
                        format_eta(eta_seconds),
                    )

                    # Log progress to database for real-time monitoring
                    segment_task_id_progress = f"{segment_task_prefix}:event=progress:count={processed:06d}"
                    log_to_db(
                        endpoint='welcome_to_the_jungle',
                        level='INFO',
                        message=(
                            f"⏳ Segment {segment}: {processed}/{total_urls} ({pct:.1f}%) "
                            f"- {ok} OK, {ko} erreurs - {rate:.2f} req/s "
                            f"- Écoulé: {format_eta(elapsed)} - ETA: {format_eta(eta_seconds)}"
                        ),
                        task_id=segment_task_id_progress,
                        duration_sec=round(batch_duration, 2),
                        records_count=processed,
                        error_count=ko,
                        extra_metadata={
                            'segment': segment,
                            'processed': processed,
                            'total': total_urls,
                            'percent': round(pct, 1),
                            'ok': ok,
                            'ko': ko,
                            'rate_req_per_sec': round(rate, 2),
                            'eta_seconds': round(eta_seconds, 1),
                            'batch_duration': round(batch_duration, 2),
                            'elapsed_total': round(elapsed, 2),
                            'active_futures': len(active_futures),
                            'run_id': run_id
                        }
                    )

                    # Update checkpoint time
                    last_checkpoint_time = current_time

            # Resoumettre de nouveaux URLs pour maintenir SUBMIT_BATCH futures actifs.
            # C'est le cœur du sliding window : on ne soumet de nouveaux travaux
            # qu'après avoir traité et libéré les futures terminés.
            fill_queue()

    # Flush buffer at the end if there are any remaining records that haven't been written yet.
    flush_buffer()

    total_elapsed = time.time() - start_time
    logger.info(
        "Done segment=%s | processed=%d | written=%d | ok=%d ko=%d | total_time=%s",
        segment, processed, total_written, ok, ko, format_eta(total_elapsed)
    )

    # Log final segment completion to database
    segment_task_id_end = f"{segment_task_prefix}:event=end"
    log_to_db(
        endpoint='welcome_to_the_jungle',
        level='INFO',
        message=(
            f"✅ Segment {segment} terminé: {total_written} offres importées "
            f"({ok} OK, {ko} erreurs) - {format_eta(total_elapsed)}"
        ),
        task_id=segment_task_id_end,
        duration_sec=round(total_elapsed, 2),
        records_count=total_written,
        error_count=ko,
        extra_metadata={
            'segment': segment,
            'processed': processed,
            'written': total_written,
            'ok': ok,
            'ko': ko,
            'throughput': round(total_written / total_elapsed, 2) if total_elapsed > 0 else 0,
            'run_id': run_id
        }
    )

    return {
        "segment": segment,
        "processed": processed,
        "written": total_written,
        "ok": ok,
        "ko": ko,
        "elapsed_s": total_elapsed
    }
