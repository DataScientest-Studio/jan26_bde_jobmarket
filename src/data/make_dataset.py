
import os
import re
from typing import Any, Dict, List, Iterable, Optional

import pandas as pd

from src.config.env import require_env, get_project_root, load_project_env
load_project_env()  # safe à rappeler (idempotent)

from src.storage import get_storage_from_env

OUTPUT_KEY = "datasets/rome_dataset.parquet"  # relative to gold/

MIN_CLASS_COUNT = int(os.getenv("MIN_CLASS_COUNT", "50"))
MAX_COMPETENCES = int(os.getenv("MAX_COMPETENCES", "25"))

# For MVP  we take all offers on all run.
# Otherwise use speciific dt or runid
BRONZE_PREFIX = os.getenv("BRONZE_OFFERS_PREFIX", "offers/dt=2026-02-13")

_whitespace_re = re.compile(r"\s+")

def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = _whitespace_re.sub(" ", s)
    return s.strip()

def extract_competences(record: Dict[str, Any], max_items: int = 25) -> List[str]:
    raw = record.get("competences", [])
    out: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                lib = clean_text(item.get("libelle"))
                if lib:
                    out.append(lib)

    # dedup keep order
    seen = set()
    dedup = []
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(x)
    return dedup[:max_items]

def build_text_field(record: Dict[str, Any]) -> str:
    title = clean_text(record.get("intitule"))
    desc = clean_text(record.get("description"))
    comps = extract_competences(record, MAX_COMPETENCES)

    parts = []
    if title:
        parts.append(f"[TITRE] {title}")
    if desc:
        parts.append(f"[DESC] {desc}")
    if comps:
        parts.append("[COMP] " + " ".join(comps))

    return "\n".join(parts).strip()

def iter_bronze_offers(storage) -> Iterable[Dict[str, Any]]:
    keys = storage.list_keys(BRONZE_PREFIX)
    # ne garder que les jsonl
    keys = [k for k in keys if k.endswith(".jsonl")]
    for key in keys:
        for rec in storage.read_jsonl(key):
            yield rec

def write_parquet(storage, key: str, df: pd.DataFrame) -> None:
    # Si vous avez déjà ajouté write_parquet au storage, c’est parfait.
    if hasattr(storage, "write_parquet"):
        storage.write_parquet(key, df)
        return

    raise RuntimeError(
        "write_parquet() not implemented in storage backend. "
        "Ajoute write_parquet() à storage.py (LocalStorage + S3Storage)."
    )

def main():
    # Read from bronze/france_travail
    storage_bronze = get_storage_from_env("bronze", "france_travail")
    # Write to gold
    storage_gold = get_storage_from_env("gold")

    print(f" make_dataset — reading from prefix: {BRONZE_PREFIX}")
    rows = []

    total = 0
    kept = 0
    iterable_bronze_offers = iter_bronze_offers(storage_bronze)
    for rec in iterable_bronze_offers:
        total += 1

        rome = rec.get("romeCode")
        if not rome:
            continue

        text = build_text_field(rec)
        if not text:
            continue

        rows.append(
            {"id": rec.get("id"), "text": text, "romeCode": rome}
        )
        kept += 1

        if total % 5000 == 0:
            print(f"\r📦 Records seen: {total} | Rows kept: {kept}", end="")        

    # Convert to DataFrame for easier processing and filtering
    df = pd.DataFrame(rows)
    print(f"📦 Records seen (job offers) : {total}")
    print(f"📊 Rows kept (pre-filter): {len(df)}")

    if len(df) == 0:
        raise RuntimeError("No training rows produced. Check your bronze path and fields.")

    counts = df["romeCode"].value_counts()
    print(f"📌 Count ROME Codes without filtering : {counts} (ROME codes repository contains 1584 codes)")

    # Filter out classes with too few examples to ensure a minimum viable dataset for training
    keep_codes = counts[counts >= MIN_CLASS_COUNT].index
    df = df[df["romeCode"].isin(keep_codes)].reset_index(drop=True)

    print(f"📌 Classes kept (# elements >= {MIN_CLASS_COUNT}): {len(keep_codes)} / 1584 ROME codes")
    print(f"📊 Rows kept (post-filter): {len(df)}")

    print(f"💾 Writing dataset to: {OUTPUT_KEY}")
    write_parquet(storage_gold, OUTPUT_KEY, df)

    print("✅ make_dataset — done")

if __name__ == "__main__":
    main()
