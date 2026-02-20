"""
Bronze ingestion for France Travail offers API.

Strategy:
- Retrieve ROME codes (code + libelle).
- Probe global totals per codeROME using Content-Range.
- If total <= MAX_RETRIEVABLE:
    -> Extract with range pagination (0-149, 150-299, ...).
- If total > MAX_RETRIEVABLE:
    -> Extract with backward fixed time windows (WINDOW_DAYS).
    -> Stop when a window returns 0.
    -> If a window total still exceeds MAX_RETRIEVABLE, apply binary time split.
- Persist offer payloads as JSONL under a partitioned folder structure.
- Persist run metadata (parameters, strategy, stats) as JSON.

Storage layout:
data/france_travail/
  bronze/
    offers/
      dt=YYYY-MM-DD/
       run_id=YYYYMMDDTHHMMSSZ/
        code_rome=XXXX/
            segment=global/
                part-000001.jsonl
            segment=minCreationDate=..._maxCreationDate=.../
                part-000001.jsonl    metadata/
      runs/
        run_id=YYYYMMDDTHHMMSSZ/
          run.json
"""

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from termcolor import colored

from src.ingest.france_travail_client import FranceTravailClient


# -----------------------------
# Global constants
# -----------------------------

# Content-Range format example: "offres 0-149/8423"
CONTENT_RANGE_RE = re.compile(r"offres\s+(\d+)-(\d+)/(\d+)", re.IGNORECASE)

# API range pagination constraints imply a maximum retrievable per query-filter set
MAX_RETRIEVABLE = 3150

# Offers search endpoint
OFFERS_SEARCH_PATH = "/partenaire/offresdemploi/v2/offres/search"

# ROME metiers endpoint
ROME_METIERS_URL = "https://api.francetravail.io/partenaire/rome-metiers/v1/metiers/metier"


# -----------------------------
# Data models
# -----------------------------

@dataclass(frozen=True)
class RomeItem:
    code: str
    libelle: str


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class WindowStat:
    start: str
    end: str
    total: int


# -----------------------------
# Filesystem utilities
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_dt_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


# -----------------------------
# API helpers
# -----------------------------

def fs_safe(value: str) -> str:
    # Replace Windows-invalid characters: < > : " / \ | ? * and control chars
    value = value.strip()
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "-", value)

def parse_total_from_content_range(value: str) -> int:
    match = CONTENT_RANGE_RE.search(value or "")
    if not match:
        return 0
    return int(match.group(3))


def to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def probe_total(client: FranceTravailClient, path: str, params: Dict[str, Any]) -> int:
    probe_params = dict(params)
    probe_params["range"] = "0-0"
    resp = client.request("GET", path, params=probe_params)
    return parse_total_from_content_range(resp.headers.get("Content-Range", ""))


def get_rome_metiers(client: FranceTravailClient) -> List[RomeItem]:
    data = client.get(ROME_METIERS_URL, params={"champs": "code,libelle"})
    unique: Dict[str, str] = {}
    for item in data:
        code = item.get("code")
        libelle = item.get("libelle")
        if code and libelle:
            unique[code] = libelle
    return [RomeItem(code=c, libelle=unique[c]) for c in sorted(unique.keys())]


# -----------------------------
# Console utilities
# -----------------------------

def print_rome_line(code: str, total: int, libelle: str) -> None:
    line = f"rome={code}\t\ttotal={total}\t\tlibelle={libelle}"
    if total > MAX_RETRIEVABLE:
        print(colored(line, "red"))
    else:
        print(line)


# -----------------------------
# Window generation and splitting
# -----------------------------

def build_fixed_windows_backward(now_utc: datetime, window_days: int, max_windows: int) -> List[Window]:
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

    while stack:
        s, e = stack.pop()
        params = dict(base_params)
        params["codeROME"] = code_rome
        params["minCreationDate"] = to_iso_z(s)
        params["maxCreationDate"] = to_iso_z(e)

        total = probe_total(client, OFFERS_SEARCH_PATH, params)
        window_seconds = int((e - s).total_seconds())

        if total <= MAX_RETRIEVABLE or window_seconds <= min_seconds:
            parts.append(WindowStat(start=params["minCreationDate"], end=params["maxCreationDate"], total=total))
            continue

        mid = s + (e - s) / 2
        left = (s, mid)
        right = (mid + timedelta(seconds=1), e)
        stack.append(right)
        stack.append(left)

    parts.sort(key=lambda w: (w.start, w.end))
    return parts


# -----------------------------
# Extraction functions
# -----------------------------

def extract_and_store_by_range(
    client: FranceTravailClient,
    out_dir: Path,
    code_rome: str,
    base_params: Dict[str, Any],
    page_size: int = 150,
) -> Dict[str, Any]:
    
    ensure_dir(out_dir)

    start = 0
    part_index = 0
    total_written = 0
    calls = 0

    while True:
        end = min(start + page_size - 1, 3149)
        params = dict(base_params)
        params["codeROME"] = code_rome
        params["range"] = f"{start}-{end}"

        payload = client.get(OFFERS_SEARCH_PATH, params=params)
        results = payload.get("resultats") or []

        calls += 1

        if not results:
            break

        part_index += 1
        file_path = out_dir / f"part-{part_index:06d}.jsonl"
        written = write_jsonl(file_path, results)
        total_written += written

        if written < page_size or end >= 3149:
            break

        start = end + 1

    return {
        "mode": "range",
        "calls": calls,
        "files": part_index,
        "written": total_written,
    }


def extract_and_store_by_windows(
    client: FranceTravailClient,
    out_dir: Path,
    code_rome: str,
    base_params: Dict[str, Any],
    total_global: int,
    window_days: int,
    max_windows: int,
    binary_split_min_seconds: int,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    now_utc = datetime.now(timezone.utc)
    windows = build_fixed_windows_backward(now_utc, window_days=window_days, max_windows=max_windows)

    part_index = 0
    total_written = 0
    calls = 0
    windows_used = 0
    sum_windows = 0

    for w in windows:
        params = dict(base_params)
        params["codeROME"] = code_rome
        params["minCreationDate"] = to_iso_z(w.start)
        params["maxCreationDate"] = to_iso_z(w.end)

        total_window = probe_total(client, OFFERS_SEARCH_PATH, params)
        calls += 1
        windows_used += 1

        line = f"  window={params['minCreationDate']}..{params['maxCreationDate']}\t\ttotal={total_window}"
        if total_window > MAX_RETRIEVABLE:
            print(colored(line, "red"))
        else:
            print(line)

        if total_window == 0:
            break

        # If window fits, extract by range within that date window
        if total_window <= MAX_RETRIEVABLE:

            safe_min = fs_safe(params["minCreationDate"])
            safe_max = fs_safe(params["maxCreationDate"])

            segment_value = f"minCreationDate={safe_min}_maxCreationDate={safe_max}"
            segment_dir = out_dir / f"segment={segment_value}"

            res = extract_and_store_by_range(
                client=client,
                out_dir = segment_dir,
                code_rome=code_rome,
                base_params={
                    **base_params,
                    "minCreationDate": params["minCreationDate"],
                    "maxCreationDate": params["maxCreationDate"],
                },
                page_size=150,
            )
            calls += res["calls"]
            part_index += res["files"]
            total_written += res["written"]
            sum_windows += total_window
            if sum_windows >= total_global:
                break
            continue

        # If window exceeds limit, split window and extract each subwindow
        parts = split_window_binary(
            client=client,
            code_rome=code_rome,
            base_params=base_params,
            window=w,
            min_seconds=binary_split_min_seconds,
        )

        print(colored(f"  split=binary\t\tparts={len(parts)}", "red"))

        for p in parts:
            sub_line = f"    subwindow={p.start}..{p.end}\t\ttotal={p.total}"
            if p.total > MAX_RETRIEVABLE:
                print(colored(sub_line, "red"))
            else:
                print(sub_line)

            if p.total == 0:
                continue

            safe_start = fs_safe(p.start)
            safe_end = fs_safe(p.end)

            segment_value = f"minCreationDate={safe_start}_maxCreationDate={safe_end}"
            segment_dir = out_dir / f"segment={segment_value}"

            res = extract_and_store_by_range(
                client=client,
                out_dir=segment_dir,
                code_rome=code_rome,
                base_params={
                    **base_params,
                    "minCreationDate": p.start,
                    "maxCreationDate": p.end,
                },
                page_size=150,
            )
            calls += res["calls"]
            part_index += res["files"]
            total_written += res["written"]
            sum_windows += p.total
            if sum_windows >= total_global:
                break

        if sum_windows >= total_global:
            break

    return {
        "mode": "windows",
        "window_days": window_days,
        "windows_used": windows_used,
        "sum_windows": sum_windows,
        "calls": calls,
        "files": part_index,
        "written": total_written,
    }

# -----------------------------
# Markdown reporting
# -----------------------------
def generate_markdown_report(
    run_file: Path,
    run_payload: Dict[str, Any],
    per_rome_stats: List[Dict[str, Any]],
) -> None:
    md_path = run_file.parent / "run.md"

    totals = [x["total_global"] for x in per_rome_stats]
    over = [x for x in per_rome_stats if x["total_global"] > MAX_RETRIEVABLE]

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# France Travail Bronze Ingestion Report\n\n")
        f.write(f"- run_id: `{run_payload['run_id']}`\n")
        f.write(f"- dt: `{run_payload['dt']}`\n\n")

        f.write("## Parameters\n\n")
        f.write(f"- MAX_RETRIEVABLE: {MAX_RETRIEVABLE}\n")
        f.write(f"- WINDOW_DAYS: {run_payload['params']['window_days']}\n")
        f.write(f"- MAX_WINDOWS: {run_payload['params']['max_windows']}\n")
        f.write(f"- BINARY_SPLIT_MIN_SECONDS: {run_payload['params']['binary_split_min_seconds']}\n\n")

        f.write("## Global Stats\n\n")
        f.write(f"- ROME processed: {len(per_rome_stats)}\n")
        f.write(f"- Over limit: {len(over)}\n")
        f.write(f"- Min total_global: {min(totals) if totals else 0}\n")
        f.write(f"- Max total_global: {max(totals) if totals else 0}\n\n")

        f.write("## Per ROME Summary\n\n")
        f.write("| codeROME | libelle | total_global | mode | written | calls |\n")
        f.write("|---|---|---:|---|---:|---:|\n")
        for x in sorted(per_rome_stats, key=lambda a: a["total_global"], reverse=True):
            f.write(
                f"| {x['code']} | {x['libelle']} | {x['total_global']} | {x['mode']} | {x['written']} | {x['calls']} |\n"
            )

        if over:
            f.write("\n## Over-Limit Details\n\n")
            f.write("| codeROME | total_global | window_days | sum_windows | windows_used |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for x in sorted(over, key=lambda a: a["total_global"], reverse=True):
                f.write(
                    f"| {x['code']} | {x['total_global']} | {x.get('window_days', 0)} | {x.get('sum_windows', 0)} | {x.get('windows_used', 0)} |\n"
                )


# -----------------------------
# Main entrypoint
# -----------------------------

def main() -> None:
    client = FranceTravailClient()

    data_root = Path(os.getenv("FT_DATA_DIR", "data/france_travail"))
    bronze_root = data_root / "bronze"

    offers_root = bronze_root / "offers"
    runs_root = bronze_root / "metadata" / "runs"

    ensure_dir(offers_root)
    ensure_dir(runs_root)

    dt = utc_dt_str()
    run_id = utc_run_id()

    window_days = int(os.getenv("FT_WINDOW_DAYS", "7"))
    max_windows = int(os.getenv("FT_MAX_WINDOWS", "260"))
    binary_split_min_seconds = int(os.getenv("FT_BINARY_SPLIT_MIN_SECONDS", "3600"))
    max_rome_codes = int(os.getenv("FT_MAX_ROME_CODES", "0"))

    run_dir = runs_root / f"run_id={run_id}"
    ensure_dir(run_dir)

    # Base search params for all requests
    base_params: Dict[str, Any] = {
        "sort": "1",
    }

    rome_items = get_rome_metiers(client)
    if max_rome_codes > 0:
        rome_items = rome_items[:max_rome_codes]

    per_rome_stats: List[Dict[str, Any]] = []
    total_calls = 0
    total_written = 0
    started = time.time()

    for rome in rome_items:
        total_global = probe_total(client, OFFERS_SEARCH_PATH, {"codeROME": rome.code, **base_params})
        print_rome_line(rome.code, total_global, rome.libelle)

        out_dir = offers_root / f"dt={dt}" / f"run_id={run_id}" / f"code_rome={rome.code}"

        ensure_dir(out_dir)

        if total_global <= MAX_RETRIEVABLE:
            res = extract_and_store_by_range(
                client=client,
                out_dir=out_dir / "segment=global",
                code_rome=rome.code,
                base_params=base_params,
                page_size=150,
            )
            stat = {
                "code": rome.code,
                "libelle": rome.libelle,
                "total_global": total_global,
                "mode": res["mode"],
                "calls": res["calls"],
                "written": res["written"],
            }
        else:
            res = extract_and_store_by_windows(
                client=client,
                out_dir=out_dir,
                code_rome=rome.code,
                base_params=base_params,
                total_global=total_global,
                window_days=window_days,
                max_windows=max_windows,
                binary_split_min_seconds=binary_split_min_seconds,
            )
            stat = {
                "code": rome.code,
                "libelle": rome.libelle,
                "total_global": total_global,
                "mode": res["mode"],
                "window_days": res["window_days"],
                "windows_used": res["windows_used"],
                "sum_windows": res["sum_windows"],
                "calls": res["calls"],
                "written": res["written"],
            }

        per_rome_stats.append(stat)
        total_calls += stat["calls"]
        total_written += stat["written"]

    elapsed = time.time() - started

    run_payload = {
        "run_id": run_id,
        "dt": dt,
        "params": {
            "window_days": window_days,
            "max_windows": max_windows,
            "binary_split_min_seconds": binary_split_min_seconds,
            "max_rome_codes": max_rome_codes,
            "max_retrievable": MAX_RETRIEVABLE,
        },
        "stats": {
            "rome_processed": len(per_rome_stats),
            "calls": total_calls,
            "written": total_written,
            "elapsed_s": round(elapsed, 2),
        },
        "storage": {
            "data_root": str(data_root),
            "offers_root": str(offers_root),
            "run_dir": str(run_dir),
        },
    }

    run_file = run_dir / "run.json"
    write_json(run_file, run_payload)
    generate_markdown_report(run_file, run_payload, per_rome_stats)

    print(f"\nrun_id={run_id}")
    print(f"calls={total_calls}\twritten={total_written}\telapsed_s={elapsed:.2f}")
    print(f"run_json={run_file}")
    print(f"run_md={run_dir / 'run.md'}")


if __name__ == "__main__":
    main()
