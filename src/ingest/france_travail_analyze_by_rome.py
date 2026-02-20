"""
ROME extraction strategy analysis for France Travail API.

This script:
1. Retrieves all ROME codes.
2. Computes the global number of offers per ROME code.
3. If total <= MAX_RETRIEVABLE:
       -> Simple extraction strategy (no split required).
4. If total > MAX_RETRIEVABLE:
       -> Applies backward fixed time windows (e.g., 7 days).
       -> If a window still exceeds MAX_RETRIEVABLE,
          a binary time split is applied inside that window.
5. Generates a Markdown report summarizing results.
"""

import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from termcolor import colored
from src.ingest.france_travail_client import FranceTravailClient


# ==============================
# Global Configuration Variables
# ==============================

# Regular expression used to extract total result count from Content-Range header
# Example header format: "offres 0-149/8423"
CONTENT_RANGE_RE = re.compile(r"offres\s+(\d+)-(\d+)/(\d+)", re.IGNORECASE)

# Maximum number of offers retrievable via range pagination
# API constraint: p <= 3000 and d <= 3149 => maximum 3150 results
MAX_RETRIEVABLE = 3150

# Output Markdown report file
OUTPUT_FILE = "rome_volume_analysis.md"

# Default size of backward time window in days
# Example: 7 means windows like [now-7d, now], [now-14d, now-7d], etc.
WINDOW_DAYS_DEFAULT = 7

# Maximum number of backward windows generated
# 260 windows of 7 days ≈ 5 years of historical coverage
MAX_WINDOWS_DEFAULT = 260

# Minimum duration (in seconds) allowed for binary split recursion
# Prevents infinite splitting of extremely dense periods
BINARY_SPLIT_MIN_SECONDS_DEFAULT = 3600  # 1 hour


# ==============================
# Data Models
# ==============================

@dataclass(frozen=True)
class RomeItem:
    """Represents a ROME code and its label."""
    code: str
    libelle: str


@dataclass(frozen=True)
class WindowStat:
    """Represents a time window and its associated offer count."""
    start: str
    end: str
    total: int


@dataclass(frozen=True)
class RomeStrategyResult:
    """
    Final strategy result for a ROME code.

    total_global: total number of offers (no date filter)
    strategy: strategy applied (simple or window-based)
    windows: generated time windows
    sum_windows: sum of offers across all generated windows
    match_global: whether sum_windows == total_global
    """
    code: str
    libelle: str
    total_global: int
    strategy: str
    window_days: int
    windows: Tuple[WindowStat, ...]
    sum_windows: int
    match_global: bool


# ==============================
# Utility Functions
# ==============================

def parse_total_from_content_range(value: str) -> int:
    """
    Extracts total result count from Content-Range header.
    """
    match = CONTENT_RANGE_RE.search(value or "")
    if not match:
        return 0
    return int(match.group(3))


def to_iso_z(dt: datetime) -> str:
    """
    Converts datetime to ISO 8601 UTC string.
    """
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ==============================
# API Interaction
# ==============================

def get_rome_metiers(client: FranceTravailClient) -> List[RomeItem]:
    """
    Retrieves all ROME codes and labels from API.
    """
    data = client.get(
        "https://api.francetravail.io/partenaire/rome-metiers/v1/metiers/metier",
        params={"champs": "code,libelle"},
    )

    unique: Dict[str, str] = {}
    for item in data:
        code = item.get("code")
        libelle = item.get("libelle")
        if code and libelle:
            unique[code] = libelle

    return [RomeItem(code=c, libelle=unique[c]) for c in sorted(unique.keys())]


def probe_total(client: FranceTravailClient, path: str, params: Dict[str, str]) -> int:
    """
    Performs a lightweight query (range=0-0) to retrieve only total count.
    """
    probe_params = dict(params)
    probe_params["range"] = "0-0"
    resp = client.request("GET", path, params=probe_params)
    return parse_total_from_content_range(resp.headers.get("Content-Range", ""))


# ==============================
# Time Window Logic
# ==============================

def build_fixed_windows_backward(
    now_utc: datetime,
    window_days: int,
    max_windows: int,
) -> List[Tuple[datetime, datetime]]:
    """
    Generates fixed-size backward windows from now.

    Example (window_days=7):
    [now-7d, now]
    [now-14d, now-7d]
    ...
    """
    windows = []
    end = now_utc
    delta = timedelta(days=window_days)

    for _ in range(max_windows):
        start = end - delta
        windows.append((start, end))
        end = start

    return windows


def split_window_binary(
    client: FranceTravailClient,
    path: str,
    base_params: Dict[str, str],
    code_rome: str,
    start: datetime,
    end: datetime,
    min_seconds: int,
) -> List[WindowStat]:
    """
    Applies recursive binary split on a window if it exceeds MAX_RETRIEVABLE.
    """
    stack = [(start, end)]
    parts: List[WindowStat] = []

    while stack:
        s, e = stack.pop()

        params = dict(base_params)
        params["codeROME"] = code_rome
        params["minCreationDate"] = to_iso_z(s)
        params["maxCreationDate"] = to_iso_z(e)

        total = probe_total(client, path, params)
        window_seconds = int((e - s).total_seconds())

        if total <= MAX_RETRIEVABLE or window_seconds <= min_seconds:
            parts.append(WindowStat(start=params["minCreationDate"], end=params["maxCreationDate"], total=total))
            continue

        mid = s + (e - s) / 2
        stack.append((mid + timedelta(seconds=1), e))
        stack.append((s, mid))

    return parts


# ==============================
# Core Analysis Logic
# ==============================

def print_rome_line(code: str, total: int, libelle: str) -> None:
    """
    Prints ROME summary line.
    Red color is used when total exceeds MAX_RETRIEVABLE.
    """
    line = f"rome={code}\t\ttotal={total}\t\tlibelle={libelle}"
    if total > MAX_RETRIEVABLE:
        print(colored(line, "red"))
    else:
        print(line)


def analyse_code(
    client: FranceTravailClient,
    path: str,
    base_params: Dict[str, str],
    rome: RomeItem,
    window_days: int,
    max_windows: int,
    binary_split_min_seconds: int,
) -> RomeStrategyResult:

    global_params = dict(base_params)
    global_params["codeROME"] = rome.code

    total_global = probe_total(client, path, global_params)
    print_rome_line(rome.code, total_global, rome.libelle)

    # No split required
    if total_global <= MAX_RETRIEVABLE:
        return RomeStrategyResult(
            code=rome.code,
            libelle=rome.libelle,
            total_global=total_global,
            strategy="codeROME",
            window_days=0,
            windows=tuple(),
            sum_windows=total_global,
            match_global=True,
        )

    # Window-based strategy required
    now_utc = datetime.now(timezone.utc)
    windows_raw = build_fixed_windows_backward(now_utc, window_days, max_windows)

    collected: List[WindowStat] = []
    sum_windows = 0

    for (w_start, w_end) in windows_raw:
        params = dict(base_params)
        params["codeROME"] = rome.code
        params["minCreationDate"] = to_iso_z(w_start)
        params["maxCreationDate"] = to_iso_z(w_end)

        total_window = probe_total(client, path, params)

        if total_window == 0:
            break

        if total_window <= MAX_RETRIEVABLE:
            collected.append(WindowStat(start=params["minCreationDate"], end=params["maxCreationDate"], total=total_window))
            sum_windows += total_window
            if sum_windows >= total_global:
                break
            continue

        split_parts = split_window_binary(
            client,
            path,
            base_params,
            rome.code,
            w_start,
            w_end,
            binary_split_min_seconds,
        )

        collected.extend(split_parts)
        sum_windows += sum(p.total for p in split_parts)
        if sum_windows >= total_global:
            break

    return RomeStrategyResult(
        code=rome.code,
        libelle=rome.libelle,
        total_global=total_global,
        strategy=f"codeROME × backward_windows({window_days}d)",
        window_days=window_days,
        windows=tuple(collected),
        sum_windows=sum_windows,
        match_global=(sum_windows == total_global),
    )
