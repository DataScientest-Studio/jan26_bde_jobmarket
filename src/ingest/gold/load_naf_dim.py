"""NAF dimension loader: builds gold.dim_naf from INSEE CSV files.

Source — storage layer bronze/insee (S3/MinIO: bronze/insee/, local: data/bronze/insee/):
  - int_courts_naf_rev_2.csv : labels for all 5 NAF levels (UTF-8 with BOM, sep=';')

File is read via get_storage_from_env("bronze", "insee") using read_bytes() + BytesIO,
so the same code works against both local filesystem and MinIO/S3.

All hierarchy codes are derived directly from the CSV:
  - div_code    : first 2 chars of naf_code (e.g. '62.01Z' → '62')
  - section_code: propagated from 'SECTION X' rows that precede each division block

Structure extracted from int_courts_naf_rev_2.csv by code pattern:
  SECTION X  → NIV1 section    (single letter + label, e.g. 'J' / 'Information et communication')
  XX         → NIV2 division   (2-digit code + label, e.g. '62')
  XX.X       → NIV3 groupe     (code + label, e.g. '62.0')
  XX.XX      → NIV4 classe     (code + label, e.g. '62.01')
  XX.XXY     → NIV5 sous-classe (leaf, stored in naf_code, e.g. '62.01Z')

The resulting dim_naf is flat (one row per sous-classe) with all ancestor levels
denormalized. This is intentional: GROUP BY at any level requires no extra JOIN.

Note: accented characters read correctly from the UTF-8 CSV; terminal display
may show substitution glyphs on Windows consoles — the stored data is correct.

---------------------------------------------------------------------
Design: flat dimension (star schema)
---------------------------------------------------------------------

The NAF classification has 5 levels (section → division → groupe → classe →
sous-classe). A snowflake design would split these into 5 linked tables; a
star schema collapses them into one flat row per sous-classe.

Star schema is chosen here for the same reasons as dim_geo:

  Pros
  - GROUP BY at any NAF level (section, division, groupe, classe) requires
    exactly one JOIN from fact_offre_emploi to dim_naf — no chain of JOINs.
  - Grafana panels stay simple: SELECT section_label, COUNT(*) ... JOIN dim_naf
    works regardless of the aggregation granularity.

  Cons
  - Label redundancy: section_label 'Information et communication' is repeated
    for all ~80 sous-classes in section J. Acceptable: dim_naf has 732 rows.

Indexing strategy: no indexes needed on dim_naf columns (732 rows — full scan
is faster than an index lookup). The relevant index is on fact_offre_emploi.naf_key.

Usage:
    python -m src.ingest.gold.load_naf_dim
"""

from __future__ import annotations

import io
import logging
import os

import pandas as pd

from src.config.env import load_project_env
from src.storage.storage import get_storage_from_env

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None

logger = logging.getLogger(__name__)

# S3/MinIO: bronze/insee/
# Local:    data/bronze/insee/
_INSEE_STORAGE_LAYER = "bronze"
_INSEE_STORAGE_SOURCE = "insee"
_CSV_KEY = "int_courts_naf_rev_2.csv"


def _build_naf_frame() -> pd.DataFrame:
    """Build a flat dim_naf DataFrame with all 5 hierarchy levels."""
    storage = get_storage_from_env(_INSEE_STORAGE_LAYER, _INSEE_STORAGE_SOURCE)

    raw = pd.read_csv(
        io.BytesIO(storage.read_bytes(_CSV_KEY)),
        sep=";", encoding="utf-8-sig", dtype=str,
    )
    raw.columns = ["ligne", "code", "libelle_long", "libelle_65", "libelle_40"]
    raw["code"] = raw["code"].fillna("").str.strip()
    raw["libelle_long"] = raw["libelle_long"].fillna("").str.strip()
    raw["libelle_40"] = raw["libelle_40"].fillna("").str.strip()

    def by_pattern(pat: str) -> pd.DataFrame:
        return raw[raw["code"].str.match(pat, na=False)][["code", "libelle_long"]].copy()

    # NIV1 — Section: code like 'SECTION A'
    sections = by_pattern(r"^SECTION [A-U]$").copy()
    sections["section_code"] = sections["code"].str.extract(r"SECTION ([A-U])")
    sections = sections.rename(columns={"libelle_long": "section_label"})[["section_code", "section_label"]]

    # NIV2 — Division: 2-digit code
    divs = by_pattern(r"^\d{2}$").rename(columns={"code": "div_code", "libelle_long": "div_label"})

    # NIV3 — Groupe: XX.X
    groupes = by_pattern(r"^\d{2}\.\d$").rename(columns={"code": "groupe_code", "libelle_long": "groupe_label"})
    groupes["div_code"] = groupes["groupe_code"].str[:2]

    # NIV4 — Classe: XX.XX
    classes = by_pattern(r"^\d{2}\.\d{2}$").rename(columns={"code": "classe_code", "libelle_long": "classe_label"})
    classes["groupe_code"] = classes["classe_code"].str[:4]

    # NIV5 — Sous-classe: XX.XXY (leaf)
    sousclasses = raw[raw["code"].str.match(r"^\d{2}\.\d{2}[A-Z]$", na=False)][
        ["code", "libelle_long"]
    ].rename(columns={"code": "naf_code", "libelle_long": "naf_label"})
    sousclasses["classe_code"] = sousclasses["naf_code"].str[:5]   # e.g. '01.11'
    sousclasses["div_code"] = sousclasses["naf_code"].str[:2]      # e.g. '01'

    # div_code → section_code: propagate from CSV row order
    # (each 'SECTION X' block precedes its divisions)
    div_section = {}
    current_section = None
    for _, row in raw.iterrows():
        code = row["code"]
        if code.startswith("SECTION "):
            current_section = code.replace("SECTION ", "").strip()
        elif code and len(code) == 2 and code.isdigit() and current_section:
            div_section[code] = current_section
    div_section_df = pd.DataFrame(list(div_section.items()), columns=["div_code", "section_code"])

    # Build flat frame: sous-classe → classe → groupe → division → section
    df = sousclasses.merge(classes, on="classe_code", how="left")
    df = df.merge(groupes[["groupe_code", "groupe_label"]], on="groupe_code", how="left")
    df = df.merge(divs, on="div_code", how="left")
    df = df.merge(div_section_df, on="div_code", how="left")
    df = df.merge(sections, on="section_code", how="left")

    for col in ["naf_label", "classe_label", "groupe_label", "div_label", "section_label"]:
        df[col] = df[col].fillna("").str.strip()

    df["naf_label"] = df["naf_label"].replace("", "Unknown NAF")

    return df[[
        "naf_code", "naf_label",
        "classe_code", "classe_label",
        "groupe_code", "groupe_label",
        "div_code", "div_label",
        "section_code", "section_label",
    ]].copy()


def load_naf_dim() -> dict:
    """Upsert gold.dim_naf from local INSEE CSV/Excel files.

    Returns:
        Dict with upserted row count.
    """
    if psycopg is None:
        raise RuntimeError("psycopg is required")

    load_project_env()
    dsn = os.getenv("JOBSTORE_DSN")
    if not dsn:
        raise ValueError("JOBSTORE_DSN is not set")

    df = _build_naf_frame()
    logger.info("NAF frame built: %d sous-classes", len(df))

    bootstrap_path = os.path.join("postgres", "init", "003_gold_star_schema.sql")
    with open(bootstrap_path, "r", encoding="utf-8") as f:
        bootstrap_sql = f.read()

    upserted = 0
    with psycopg.connect(dsn, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(bootstrap_sql)
        conn.commit()

        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO gold.dim_naf (
                        naf_code, naf_label,
                        classe_code, classe_label,
                        groupe_code, groupe_label,
                        div_code, div_label,
                        section_code, section_label
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (naf_code) DO UPDATE SET
                        naf_label    = EXCLUDED.naf_label,
                        classe_code  = EXCLUDED.classe_code,
                        classe_label = EXCLUDED.classe_label,
                        groupe_code  = EXCLUDED.groupe_code,
                        groupe_label = EXCLUDED.groupe_label,
                        div_code     = EXCLUDED.div_code,
                        div_label    = EXCLUDED.div_label,
                        section_code = EXCLUDED.section_code,
                        section_label = EXCLUDED.section_label,
                        updated_at   = NOW()
                    """,
                    (
                        row["naf_code"],
                        row["naf_label"] or None,
                        row["classe_code"] or None,
                        row["classe_label"] or None,
                        row["groupe_code"] or None,
                        row["groupe_label"] or None,
                        row["div_code"] or None,
                        row["div_label"] or None,
                        row["section_code"] or None,
                        row["section_label"] or None,
                    ),
                )
                upserted += 1
        conn.commit()

    logger.info("dim_naf upserted: %d rows", upserted)
    return {"upserted": upserted}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = load_naf_dim()
    logger.info("NAF dim load completed: %s", result)


if __name__ == "__main__":
    main()
