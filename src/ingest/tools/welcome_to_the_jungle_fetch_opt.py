"""
Welcome to the Jungle — Optimized fetch utilities
===================================================

Provides optimized HTTP fetching using curl_cffi with impersonate for better anti-bot protection.
Falls back to regular requests if curl_cffi is not available.

Applies throttling (random delay AFTER each fetch, within the worker) to avoid bot detection
while maintaining parallelism. Each worker independently respects the delay, simulating human 
reading time and preventing burst-based bot detection.

Designed to be a drop-in replacement for fetch_page() in welcome_to_the_jungle_common.py
with identical signature and return type.
"""

import logging
import os
import random
import threading
import time
from typing import Optional

logger = logging.getLogger("wttj.ingest.bronze.fetch_opt")


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


LOG_FETCH_WITH_WORKER = _as_bool(os.getenv("WTTJ_LOG_FETCH_WITH_WORKER", "false"), default=False)
LOG_FIRST_FETCHES = int(os.getenv("WTTJ_LOG_FIRST_FETCHES", "0"))
ADAPTIVE_429_COOLDOWN = _as_bool(os.getenv("WTTJ_ADAPTIVE_429_COOLDOWN", "true"), default=True)
STARTUP_SPREAD_SECONDS = float(os.getenv("WTTJ_STARTUP_SPREAD_SECONDS", "2.0"))
STARTUP_WORKERS_ESTIMATE = int(os.getenv("WTTJ_WORKERS", "8"))
GLOBAL_START_SPACING_SECONDS = float(os.getenv("WTTJ_GLOBAL_START_SPACING_SECONDS", "0.0"))
GLOBAL_429_COOLDOWN_SECONDS = float(os.getenv("WTTJ_429_COOLDOWN_SECONDS", "15"))
GLOBAL_429_COOLDOWN_MAX_SECONDS = float(os.getenv("WTTJ_429_COOLDOWN_MAX_SECONDS", "45"))
GLOBAL_429_COOLDOWN_STEP_SECONDS = float(os.getenv("WTTJ_429_COOLDOWN_STEP_SECONDS", "5"))
GLOBAL_429_COOLDOWN_DECAY_SUCCESS = int(os.getenv("WTTJ_429_COOLDOWN_DECAY_SUCCESS", "120"))
GLOBAL_429_ESCALATION_MIN_INTERVAL_SECONDS = float(os.getenv("WTTJ_429_ESCALATION_MIN_INTERVAL_SECONDS", "10"))
_log_counter_lock = threading.Lock()
_log_counter = 0
_cooldown_lock = threading.Lock()
_global_cooldown_until = 0.0
_current_cooldown_seconds = GLOBAL_429_COOLDOWN_SECONDS
_success_since_429 = 0
_last_cooldown_escalation_ts = 0.0
_startup_seen_threads: set[int] = set()
_startup_base_ts = 0.0
_startup_slot_counter = 0
_global_next_start_ts = 0.0


def _apply_startup_stagger(worker_name: str, worker_id: int) -> None:
    """Apply one-time deterministic startup delay per worker to avoid first-wave burst."""
    if STARTUP_SPREAD_SECONDS <= 0:
        return

    should_stagger = False
    slot = 0
    base_ts = 0.0
    with _cooldown_lock:
        global _startup_base_ts, _startup_slot_counter
        if worker_id not in _startup_seen_threads:
            _startup_seen_threads.add(worker_id)
            should_stagger = True
            if _startup_base_ts == 0.0:
                _startup_base_ts = time.time()
            slot = _startup_slot_counter
            _startup_slot_counter += 1
            base_ts = _startup_base_ts

    if should_stagger:
        workers = max(1, STARTUP_WORKERS_ESTIMATE)
        if workers == 1:
            stagger = 0.0
        else:
            slot_interval = STARTUP_SPREAD_SECONDS / (workers - 1)
            target_ts = base_ts + (slot * slot_interval)
            stagger = max(0.0, target_ts - time.time())
        if LOG_FETCH_WITH_WORKER:
            logger.info("FETCH_STAGGER | worker=%s id=%s | slot=%d | sleep=%.3fs", worker_name, worker_id, slot, stagger)
        time.sleep(stagger)


def _acquire_global_start_slot(worker_name: str, worker_id: int) -> None:
    """Enforce a global spacing between request starts across all workers."""
    if GLOBAL_START_SPACING_SECONDS <= 0:
        return

    with _cooldown_lock:
        global _global_next_start_ts
        now = time.time()
        slot_ts = max(now, _global_next_start_ts)
        _global_next_start_ts = slot_ts + GLOBAL_START_SPACING_SECONDS

    wait_seconds = max(0.0, slot_ts - time.time())
    if wait_seconds > 0:
        if LOG_FETCH_WITH_WORKER:
            logger.info("FETCH_GATE | worker=%s id=%s | sleep=%.3fs", worker_name, worker_id, wait_seconds)
        time.sleep(wait_seconds)


def _should_log_fetch() -> bool:
    global _log_counter
    if not LOG_FETCH_WITH_WORKER:
        return False
    if LOG_FIRST_FETCHES <= 0:
        return True
    with _log_counter_lock:
        if _log_counter >= LOG_FIRST_FETCHES:
            return False
        _log_counter += 1
        return True


def _apply_global_cooldown_if_needed() -> None:
    now = time.time()
    with _cooldown_lock:
        wait_seconds = _global_cooldown_until - now
    if wait_seconds > 0:
        time.sleep(wait_seconds)


def _set_global_cooldown(seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    new_until = time.time() + seconds
    with _cooldown_lock:
        global _global_cooldown_until
        if new_until > _global_cooldown_until:
            _global_cooldown_until = new_until
            return seconds
        return max(0.0, _global_cooldown_until - time.time())


def _register_429_and_get_cooldown(retry_after: float) -> tuple[float, float]:
    """Return (target_cooldown_seconds, applied_wait_seconds)."""
    with _cooldown_lock:
        global _current_cooldown_seconds, _success_since_429, _global_cooldown_until, _last_cooldown_escalation_ts
        _success_since_429 = 0
        now = time.time()

        if ADAPTIVE_429_COOLDOWN:
            # Debounce escalation: increase at most once per configured interval,
            # so a burst of simultaneous 429s does not ramp cooldown too aggressively.
            if (now - _last_cooldown_escalation_ts) >= GLOBAL_429_ESCALATION_MIN_INTERVAL_SECONDS:
                _current_cooldown_seconds = min(
                    GLOBAL_429_COOLDOWN_MAX_SECONDS,
                    max(GLOBAL_429_COOLDOWN_SECONDS, _current_cooldown_seconds + GLOBAL_429_COOLDOWN_STEP_SECONDS),
                )
                _last_cooldown_escalation_ts = now
        else:
            _current_cooldown_seconds = GLOBAL_429_COOLDOWN_SECONDS

        target_cooldown = max(_current_cooldown_seconds, retry_after)
        new_until = now + target_cooldown
        if new_until > _global_cooldown_until:
            _global_cooldown_until = new_until
            applied = target_cooldown
        else:
            applied = max(0.0, _global_cooldown_until - now)

    return target_cooldown, applied


def _register_success_for_cooldown_decay() -> Optional[float]:
    """Returns new cooldown when decayed, else None."""
    if not ADAPTIVE_429_COOLDOWN:
        return None

    with _cooldown_lock:
        global _success_since_429, _current_cooldown_seconds
        _success_since_429 += 1
        if (
            _current_cooldown_seconds > GLOBAL_429_COOLDOWN_SECONDS
            and _success_since_429 >= GLOBAL_429_COOLDOWN_DECAY_SUCCESS
        ):
            _current_cooldown_seconds = max(
                GLOBAL_429_COOLDOWN_SECONDS,
                _current_cooldown_seconds - GLOBAL_429_COOLDOWN_STEP_SECONDS,
            )
            _success_since_429 = 0
            return _current_cooldown_seconds

    return None


def _retry_after_to_seconds(retry_after_value: Optional[str]) -> float:
    if not retry_after_value:
        return 0.0
    try:
        return max(0.0, float(retry_after_value))
    except Exception:
        return 0.0

# Try to import curl_cffi for better anti-bot protection
try:
    from curl_cffi import requests as cffi_requests
    USE_CFFI = True
    logger.info("curl_cffi available - using optimized fetch with browser impersonation")
except ImportError:
    import requests as cffi_requests
    USE_CFFI = False
    logger.info("curl_cffi not available - falling back to standard requests")


class FetchResult:
    """Result of fetching a page - compatible with existing FetchResult"""
    def __init__(self, url: str, ok: bool, status_code: Optional[int], html: Optional[str], error: Optional[str]):
        self.url = url
        self.ok = ok
        self.status_code = status_code
        self.html = html
        self.error = error


def fetch_page_opt(session, limiter, url: str, timeout: int = 30, delay: float = 0.0) -> FetchResult:
    """
    Optimized fetch with curl_cffi (if available) or fallback to requests.
    
    Drop-in replacement for fetch_page() with identical signature and return type.
    Delay is applied inside the worker thread after each request.
    
    Args:
        session: HTTP session (used for headers if curl_cffi not available)
        limiter: Rate limiter instance (not used in opt version, kept for compatibility)
        url: URL to fetch
        timeout: Request timeout in seconds
        delay: Base delay in seconds for worker-level random sleep (delay..delay*1.3)
    
    Returns:
        FetchResult with url, ok, status_code, html, error
    """
    worker_name = threading.current_thread().name
    worker_id = threading.get_ident()
    do_log = _should_log_fetch()
    
    try:
        _apply_startup_stagger(worker_name, worker_id)
        _apply_global_cooldown_if_needed()

        # Keep limiter active in optimized mode to avoid startup bursts.
        if limiter is not None:
            limiter.acquire()

        # Optional strict pacing between starts to avoid clustered requests.
        _acquire_global_start_slot(worker_name, worker_id)

        if do_log:
            logger.info("FETCH_START | worker=%s id=%s | url=%s", worker_name, worker_id, url)

        if USE_CFFI:
            # Use curl_cffi with Chrome impersonation for better anti-bot protection
            resp = cffi_requests.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
                },
                impersonate="chrome120",
                timeout=timeout
            )
        else:
            # Fallback to regular session-based request
            resp = session.get(url, timeout=timeout)
        
        # Handle HTTP error codes
        if resp.status_code == 429:
            retry_after = _retry_after_to_seconds(resp.headers.get("Retry-After") if hasattr(resp, "headers") else None)
            cooldown, applied = _register_429_and_get_cooldown(retry_after)
            logger.warning("[429 Rate Limited] worker=%s id=%s | url=%s", worker_name, worker_id, url)
            logger.warning("[429 Cooldown] shared cooldown=%.1fs applied=%.1fs (retry-after=%.1fs)", cooldown, applied, retry_after)
            result = FetchResult(
                url=url,
                ok=False,
                status_code=429,
                html=resp.text if hasattr(resp, 'text') else None,
                error="HTTP 429 - Rate Limited"
            )
        
        elif resp.status_code == 403:
            logger.warning("[403 Forbidden] worker=%s id=%s | url=%s", worker_name, worker_id, url)
            result = FetchResult(
                url=url,
                ok=False,
                status_code=403,
                html=resp.text if hasattr(resp, 'text') else None,
                error="HTTP 403 - Forbidden"
            )
        
        elif resp.status_code == 404:
            logger.debug("[404 Not Found] worker=%s id=%s | url=%s", worker_name, worker_id, url)
            result = FetchResult(
                url=url,
                ok=False,
                status_code=404,
                html=resp.text if hasattr(resp, 'text') else None,
                error="HTTP 404 - Not Found"
            )
        
        elif resp.status_code >= 500:
            logger.warning("[HTTP %s] Server error worker=%s id=%s | url=%s", resp.status_code, worker_name, worker_id, url)
            result = FetchResult(
                url=url,
                ok=False,
                status_code=resp.status_code,
                html=resp.text if hasattr(resp, 'text') else None,
                error=f"HTTP {resp.status_code} - Server Error"
            )
        
        elif resp.status_code >= 400:
            logger.warning("[HTTP %s] Client error worker=%s id=%s | url=%s", resp.status_code, worker_name, worker_id, url)
            result = FetchResult(
                url=url,
                ok=False,
                status_code=resp.status_code,
                html=resp.text if hasattr(resp, 'text') else None,
                error=f"HTTP {resp.status_code}"
            )
        
        else:
            # Success
            result = FetchResult(
                url=url,
                ok=True,
                status_code=resp.status_code,
                html=resp.text,
                error=None
            )
            new_cooldown = _register_success_for_cooldown_decay()
            if new_cooldown is not None:
                logger.info("[429 Cooldown] decayed shared cooldown to %.1fs after stable successes", new_cooldown)

        # Worker-level throttling must happen here, inside worker thread.
        if delay > 0:
            time.sleep(random.uniform(delay, delay * 1.3))

        if do_log:
            logger.info("FETCH_DONE | worker=%s id=%s | status=%s | url=%s", worker_name, worker_id, resp.status_code, url)
    
    except Exception as e:
        error_type = "timeout" if "timeout" in str(e).lower() else "error"
        logger.error("[FETCH %s] worker=%s id=%s | url=%s | %s: %s", error_type.upper(), worker_name, worker_id, url, type(e).__name__, e)
        result = FetchResult(
            url=url,
            ok=False,
            status_code=None,
            html=None,
            error=f"{error_type}: {type(e).__name__}"
        )
    
    return result
