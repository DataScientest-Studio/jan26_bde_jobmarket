from src.ingest.clients.france_travail_client import FranceTravailClient
from src.storage.storage import Storage
from src.utils.time_helpers import to_iso_z
from typing import Any, Dict, List, Tuple 

from src.ingest.data_models.bronze_datamodel_class import RomeItem, Window, WindowStat
import re
from datetime import datetime, timedelta, timezone
import time
from termcolor import colored
import os

FT_MAX_RETRIEVABLE = int(os.getenv("FT_MAX_RETRIEVABLE", "3150"))
CONTENT_RANGE_RE = re.compile(r"offres\s+(\d+)-(\d+)/(\d+)", re.IGNORECASE)
OFFERS_SEARCH_PATH = "/partenaire/offresdemploi/v2/offres/search"
ROME_METIERS_URL = "https://api.francetravail.io/partenaire/rome-metiers/v1/metiers/metier"

# Helper functions for the ingest pipeline.
def fs_safe(value: str) -> str:
    """ Make a string safe for filesystem keys by replacing problematic characters with '-'."""
    value = value.strip()
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "-", value)

def parse_total_from_content_range(value: str) -> int:
    """ Parse the total number of offers from the Content-Range header value. 
    Expected format: 'offres X-Y/Z'."""
    match = CONTENT_RANGE_RE.search(value or "")
    if not match:
        return 0
    return int(match.group(3))

def probe_total(client: FranceTravailClient, params: Dict[str, Any]) -> int:
    """ Probe the total number of offers for given query parameters by making a request with 'range=0-0'."""
    probe_params = dict(params)
    probe_params["range"] = "0-0"

    try:
        resp = client.request("GET", OFFERS_SEARCH_PATH, params=probe_params)
        return parse_total_from_content_range(resp.headers.get("Content-Range", ""))
    except Exception as e:
        # We return 0 to avoid stopping the pipeline on transient API/network failures
        print(colored(f"[WARN] probe_total failed for params={probe_params} error={e}", "yellow"))
        return 0    

def get_rome_metiers(client: FranceTravailClient) -> List[RomeItem]:
    """ Fetch the list of ROME metiers from the API and return a list of RomeItem(code, libelle)."""
    # In case of API failure, we want to stop the pipeline early since it's a critical part.
    try:
        data = client.get(ROME_METIERS_URL, params={"champs": "code,libelle"})
    except Exception as e:
        raise RuntimeError(f"Failed to fetch ROME metiers list: {e}") from e

    unique: Dict[str, str] = {}
    for item in data:
        code = item.get("code")
        libelle = item.get("libelle")
        if code and libelle:
            unique[code] = libelle
    return [RomeItem(code=c, libelle=unique[c]) for c in sorted(unique.keys())]


def print_rome_line(code: str, total: int, libelle: str) -> None:
    """ Print a line with ROME code, total offers, and libelle. 
    Highlight in red if total exceeds MAX_RETRIEVABLE."""

    line = f"rome={code}\t\ttotal={total}\t\tlibelle={libelle}"
    if total > FT_MAX_RETRIEVABLE:
        print(colored(line, "red"))
    else:
        print(line)


def build_fixed_windows_backward(now_utc: datetime, window_days: int, max_windows: int) -> List[Window]:
    """ Build fixed time windows going backward from now_utc, 
    each of window_days length, up to max_windows.
    
    Ex :   if now_utc = 2024-06-30T12:00:00Z, window_days=7 and max_windows=3,
           it will return windows:  
                - 2024-06-23T12:00:00Z to 2024-06-30T12:00:00Z
                - 2024-06-16T12:00:00Z to 2024-06-23T12:00:00Z
                - 2024-06-09T12:00:00Z to 2024-06-16T12:00:00Z
    Used to create time windows for querying the API when total offers exceed MAX_RETRIEVABLE,
    allowing to split the data into manageable chunks.
    """
    windows: List[Window] = []
    end = now_utc
    delta = timedelta(days=window_days)
    for _ in range(max_windows):
        start = end - delta
        windows.append(Window(start=start, end=end))
        end = start
    return windows


def split_window_binary(
    client: FranceTravailClient,
    code_rome: str,
    base_params: Dict[str, Any],
    window: Window,
    min_seconds: int,
) -> List[WindowStat]:
    stack: List[Tuple[datetime, datetime]] = [(window.start, window.end)]
    parts: List[WindowStat] = []
    """ 
        Split a time window into smaller windows using a binary search approach 
        until the total offers in each window is <= MAX_RETRIEVABLE or the window size is <= min_seconds (safe guard).

        This is used when a time window has too many offers to retrieve in one go, 
        allowing to break it down into smaller chunks that can be processed.

        Parameters:
        - client: FranceTravailClient to make API calls
        - code_rome: ROME code for the query
        - base_params: base query parameters to include in each API call
        - window: the initial time window to split
        - min_seconds: minimum window size in seconds to avoid infinite splitting on very small windows

        Returns :
            A list of WindowStat with start, end, and total offers for each sub-window.
    """

    while stack:
        # We pop the last window from the stack to process it (depth-first approach).
        start, end = stack.pop()
        params = dict(base_params)
        params["codeROME"] = code_rome
        params["minCreationDate"] = to_iso_z(start)
        params["maxCreationDate"] = to_iso_z(end)

        total = probe_total(client, params)
        window_seconds = int((end - start).total_seconds())

        # If the total offers in this window is within the retrievable limit, or if the window is too small to split further,
        # we add it to the results and continue to the next one in the stack.   
        # Otherwise, we split the window into two and add them back to the stack for processing.
        if total <= FT_MAX_RETRIEVABLE or window_seconds <= min_seconds:
            parts.append(WindowStat(start=params["minCreationDate"], end=params["maxCreationDate"], total=total))
            continue

        # Binary split: we calculate the midpoint of the window and add the two halves back to the stack for processing.    
        mid = start + (end - start) / 2
        # We add a small delta (1 second) to mid for the second half to avoid overlapping windows 
        # and ensure we don't miss any offers that might be exactly at the boundary.
        stack.append((mid + timedelta(seconds=1), end))
        stack.append((start, mid))

    # After processing all windows, we sort the results by start time before returning them.
    parts.sort(key=lambda w: (w.start, w.end))
    return parts


def offers_part_key(dt: str, run_id: str, code_rome: str, segment: str, part_index: int) -> str:
    """ Generate a storage key for a part of offers data based on the given parameters."""
    return (    
        f"offers/dt={dt}/run_id={run_id}/code_rome={code_rome}/"
        f"segment={segment}/part-{part_index:06d}.jsonl"
    )


def write_run_metadata(storage: Storage, run_id: str, payload: Dict[str, Any]) -> str:
    """ Write the run metadata JSON to storage and return the storage key.""" 
    key = f"metadata/runs/run_id={run_id}/run.json"
    storage.write_json(key, payload)
    return key

def extract_and_store_by_range(
    client: FranceTravailClient,
    storage: Storage,
    dt: str,
    run_id: str,
    code_rome: str,
    segment: str,
    base_params: Dict[str, Any],
    page_size: int = 150,
) -> Dict[str, Any]:
    """
    Extract job offers for a given ROME code using 'range' pagination and store JSONL parts.

    Parameters:
    - client: FranceTravailClient to make API calls
    - storage: Storage to write the output files    
    - dt: date string for partitioning (YYYY-MM-DD)
    - run_id: unique identifier for this run
    - code_rome: ROME code to query
    - segment: segment identifier for storage key (e.g. "global" or "minCreationDate=..._maxCreationDate=...")
    - base_params: base query parameters to include in each API call
    - page_size: number of offers to retrieve per API call (default 150)
    
    Improvements:
    - Retries with backoff on transient failures.
    - Tracks errors count and skipped ranges to report in run_payload (for audit/reporting).

    Returns: 
        A dictionary with stats about the extraction process, including:
        - mode: "range"
        - calls: total API calls made
        - files: number of JSONL part files written
        - written: total number of offers written
        - page_size: the page size used for pagination
        - errors: total number of API call errors encountered
        - skipped_ranges: list of ranges that were skipped after max retries, with details for reporting

    """

    start = 0
    part_index = 0
    total_written = 0
    calls = 0

    # Error tracking
    errors = 0
    skipped_ranges = []  # list of dicts: rome/segment/range/error

    max_end_index = 3149
    max_attempts = 3
    base_sleep_seconds = 2

    while True:
        # Calculate the end index for the current page. 
        # Use min to ensure we don't exceed the maximum allowed index (3149).
        end = min(start + page_size - 1, max_end_index)

        # Prepare the query parameters for this page, including the 'range' parameter for pagination.
        params = dict(base_params)
        params["codeROME"] = code_rome
        params["range"] = f"{start}-{end}"

        payload = None
        last_error_msg = None

        # Retry loop for this page
        for attempt in range(1, max_attempts + 1):
            try:
                # Make the API call to fetch this page of offers. If successful, break out of the retry loop.
                payload = client.get(OFFERS_SEARCH_PATH, params=params)
                calls += 1
                break
            except Exception as e:
                # On exception, we increment the error count and prepare for a retry with backoff.
                calls += 1
                errors += 1
                last_error_msg = str(e)

                if attempt < max_attempts:
                    sleep_s = base_sleep_seconds * attempt
                    print(colored(
                        f"[WARN] fetch failed (attempt {attempt}/{max_attempts}) "
                        f"rome={code_rome} range={start}-{end} segment={segment} "
                        f"sleep={sleep_s}s error={e}",
                        "yellow"
                    ))
                    time.sleep(sleep_s)
                else:
                    # After max attempts: record skipped range and continue
                    # Add the skipped range details to the list for reporting in the run metadata, including the last error message for context.
                    skipped_ranges.append({
                        "code_rome": code_rome,
                        "segment": segment,
                        "range_start": start,
                        "range_end": end,
                        "range": f"{start}-{end}",
                        "error": last_error_msg,
                    })
                    print(colored(
                        f"[WARN] skipping range after {max_attempts} failures "
                        f"rome={code_rome} range={start}-{end} segment={segment} error={e}",
                        "yellow"
                    ))
                    payload = None

        # If page failed after retries, skip it
        if payload is None:
            if end >= max_end_index:
                break
            start = end + 1
            continue

        results = payload.get("resultats") or []
        if not results:
            break

        part_index += 1
        key = offers_part_key(dt, run_id, code_rome, segment, part_index)

        # AVANT
        #written = storage.write_jsonl(key, results)

        import html
        clean_results = []
        for result in results:
            # Nettoie récursivement HTML entities
            def clean_html(data):
                if isinstance(data, dict):
                    return {k: clean_html(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [clean_html(v) for v in data]
                elif isinstance(data, str):
                    return html.unescape(data)  # ’ → ', é → é
                return data
            clean_results.append(clean_html(result))

        written = storage.write_jsonl(key, clean_results)
        
        total_written += written

        if written < page_size or end >= max_end_index:
            break

        start = end + 1

    return {
        "mode": "range",
        "calls": calls,
        "files": part_index,
        "written": total_written,
        "page_size": page_size,
        "errors": errors,
        "skipped_ranges": skipped_ranges,
    }


def extract_and_store_by_windows(
    client: FranceTravailClient,
    storage: Storage,
    dt: str,
    run_id: str,
    code_rome: str,
    base_params: Dict[str, Any],
    total_global: int,
    window_days: int,
    max_windows: int,
    binary_split_min_seconds: int,
) -> Dict[str, Any]:
    """ Extract job offers for a given ROME code using time windows and store JSONL parts.
    Parameters:
    - client: FranceTravailClient to make API calls
    - storage: Storage to write the output files
    - dt: date string for partitioning (YYYY-MM-DD)
    - run_id: unique identifier for this run
    - code_rome: ROME code for the job offers
    - base_params: base parameters for the API calls
    - total_global: total number of offers expected
    - window_days: number of days in each time window
    - max_windows: maximum number of time windows to process
    - binary_split_min_seconds: minimum duration in seconds for binary splitting of windows that exceed MAX_RETRIEVABLE (a guard against infinite splitting on very small windows)     

    Returns:
        A dictionary with stats about the extraction process, including:    
        - mode: "windows" to indicate the extraction method used for this ROME code to help with reporting and analysis
        - window_days: the number of days in each time window
        - max_windows: the maximum number of time windows processed
        - binary_split_min_seconds: the minimum duration in seconds for binary splitting of windows that exceed MAX_RETRIEVABLE (a guard against infinite splitting on very small windows)     
        - windows_used: the number of time windows that were actually used (some may be skipped if total_global is reached early)
        - sum_windows: the total number of offers covered by the windows used (should be <= total_global)
        - calls: total API calls made   
    """

    # We build the time windows going backward from now, with the specified number of days and maximum windows.
    now_utc = datetime.now(timezone.utc)
    windows = build_fixed_windows_backward(now_utc, window_days=window_days, max_windows=max_windows)

    # Initialize counters for the total number of offers written, API calls made, windows used, and sum of offers in windows.
    total_written = 0
    calls = 0
    windows_used = 0
    sum_windows = 0
    files = 0

    # Error tracking
    errors = 0
    skipped_ranges: List[Dict[str, Any]] = []

    # Iterate over the time windows. 
    # For each window, we prepare the query parameters with the min and max creation dates corresponding to the window.
    for w in windows:
        params = dict(base_params)
        params["codeROME"] = code_rome
        params["minCreationDate"] = to_iso_z(w.start)
        params["maxCreationDate"] = to_iso_z(w.end)

        # We probe the total number of offers in this window to decide how to proceed.
        # If the total is 0, we can skip it. If it's <= MAX_RETRIEVABLE, we can extract it directly with range pagination.
        # If it exceeds MAX_RETRIEVABLE, we need to split it further using the binary splitting method.
        total_window = probe_total(client, params)
        calls += 1
        windows_used += 1

        line = f"  window={params['minCreationDate']}..{params['maxCreationDate']}\t\ttotal={total_window}"
        if total_window > FT_MAX_RETRIEVABLE:
            print(colored(line, "red"))
        else:
            print(line)

        # If there are no offers we stop processing this window and move to the next one.
        if total_window == 0:
            break

        # If the total offers in this window is within the retrievable limit, we can extract it directly using the range-based extraction method.
        if total_window <= FT_MAX_RETRIEVABLE:
            safe_min = fs_safe(params["minCreationDate"])
            safe_max = fs_safe(params["maxCreationDate"])
            segment = f"minCreationDate={safe_min}_maxCreationDate={safe_max}"

            res = extract_and_store_by_range(
                client=client,
                storage=storage,
                dt=dt,
                run_id=run_id,
                code_rome=code_rome,
                segment=segment,
                base_params={
                    **base_params,
                    "minCreationDate": params["minCreationDate"],
                    "maxCreationDate": params["maxCreationDate"],
                },
                page_size=150,
            )
            # aggegate error stats from range extraction to report in run_payload
            errors += res.get("errors", 0)
            skipped_ranges.extend(res.get("skipped_ranges", []))            

            calls += res["calls"]
            files += res["files"]
            total_written += res["written"]
            sum_windows += total_window
            # If we've already covered the total number of offers expected for this ROME code with the windows we've processed, 
            # we can stop early to save time and API calls.
            if sum_windows >= total_global:
                break
            continue

        # If the total offers in this window exceeds the retrievable limit, we need to split it further using the binary splitting method.
        parts = split_window_binary(
            client=client,
            code_rome=code_rome,
            base_params=base_params,
            window=w,
            min_seconds=binary_split_min_seconds,
        )

        print(colored(f"  split=binary\t\tparts={len(parts)}", "red"))
        #   We iterate over the resulting sub-windows (parts) from the binary splitting. 
        # For each part, we prepare the segment identifier and extract the offers using the range-based extraction method.  
        for p in parts:
            sub_line = f"    subwindow={p.start}..{p.end}\t\ttotal={p.total}"
            # We highlight in red the sub-windows that still exceed the retrievable limit, 
            # which indicates that they were not fully extracted and may require further attention 
            # (e.g. manual review, or future improvement of the splitting logic).
            if p.total > FT_MAX_RETRIEVABLE:
                print(colored(sub_line, "red"))
            else:
                print(sub_line)

            if p.total == 0:
                continue

            # Create a segment identifier for this sub-window by making the start and end dates filesystem-safe,
            safe_start = fs_safe(p.start)
            safe_end = fs_safe(p.end)
            segment = f"minCreationDate={safe_start}_maxCreationDate={safe_end}"

            # Etract the offers for this sub-window using the range-based extraction method, and aggregate the stats and errors for reporting
            res = extract_and_store_by_range(
                client=client,
                storage=storage,
                dt=dt,
                run_id=run_id,
                code_rome=code_rome,
                segment=segment,
                base_params={
                    **base_params,
                    "minCreationDate": p.start,
                    "maxCreationDate": p.end,
                },
                page_size=150,
            )

            errors += res.get("errors", 0)
            skipped_ranges.extend(res.get("skipped_ranges", []))

            calls += res["calls"]
            files += res["files"]
            total_written += res["written"]
            sum_windows += p.total
            # After processing all sub-windows, we check if we've covered the total number of offers expected for this ROME code,        
            if sum_windows >= total_global:
                break
        # Check again if we've covered the total number of offers expected for this ROME code,        
        if sum_windows >= total_global:
            break
       
    # Finally, we return a dictionary with all the relevant stats about the extraction process for this ROME code using the windows method,
    return {
        "mode": "windows",
        "window_days": window_days,
        "windows_used": windows_used,
        "sum_windows": sum_windows,
        "calls": calls,
        "files": files,
        "written": total_written,
        "errors": errors,
        "skipped_ranges": skipped_ranges
    }
