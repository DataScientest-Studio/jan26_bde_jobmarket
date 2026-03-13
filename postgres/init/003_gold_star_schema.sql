-- ============================================================================
-- 003_gold_star_schema.sql
-- Star-schema bootstrap (staging + 3 dimensions + minimal fact table)
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS gold;

-- -----------------------------
-- Staging table (run-scoped)
-- -----------------------------
CREATE TABLE IF NOT EXISTS gold.stg_offer (
    run_id TEXT NOT NULL,
    snapshot_dt DATE NOT NULL,
    offer_id TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT,
    description TEXT,
    status TEXT,
    published_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    unpublished_at TIMESTAMPTZ,
    rome_code TEXT,
    rome_label TEXT,
    contract_normalized TEXT,
    contract_detail TEXT,
    contract_type TEXT,
    worktime TEXT,
    experience_normalized TEXT,
    experience_index INT,
    experience_detail TEXT,
    experience_level TEXT,
    experience_description TEXT,
    job_postal_code TEXT,
    job_city TEXT,
    company_name TEXT,
    company_city TEXT,
    company_postal_code TEXT,
    naf_code TEXT,
    salary_min NUMERIC(12,2),
    salary_max NUMERIC(12,2),
    salary_periodicity TEXT,
    periodicity TEXT,
    yearly_min NUMERIC(12,2),
    yearly_max NUMERIC(12,2),
    salary_min_computed NUMERIC(12,2),
    salary_max_computed NUMERIC(12,2),
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, offer_id, source)
);

CREATE INDEX IF NOT EXISTS idx_stg_offer_run_id ON gold.stg_offer(run_id);
CREATE INDEX IF NOT EXISTS idx_stg_offer_snapshot_dt ON gold.stg_offer(snapshot_dt);

-- -----------------------------
-- Dimensions scope
-- -----------------------------
-- Flat dimension: one row per NAF sous-classe (NIV5) with all 5 hierarchy levels
-- denormalized. Redundancy is intentional (star schema) — it allows GROUP BY at
-- any level (section, division, groupe, classe, sous-classe) without extra JOINs.
--
-- Source: INSEE NAF rev.2 Excel files (data/ directory).
-- Note: accented characters in labels may appear as '?' due to XLS encoding
-- limitations in the source file; codes are always correct.
CREATE TABLE IF NOT EXISTS gold.dim_naf (
    naf_key       BIGSERIAL PRIMARY KEY,
    -- NIV5 — Sous-classe (leaf, e.g. '62.01Z')
    naf_code      TEXT NOT NULL UNIQUE,
    naf_label     TEXT NOT NULL,
    -- NIV4 — Classe (e.g. '62.01')
    classe_code   TEXT,
    classe_label  TEXT,
    -- NIV3 — Groupe (e.g. '62.0')
    groupe_code   TEXT,
    groupe_label  TEXT,
    -- NIV2 — Division (e.g. '62')
    div_code      TEXT,
    div_label     TEXT,
    -- NIV1 — Section (e.g. 'J')
    section_code  TEXT,
    section_label TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO gold.dim_naf (naf_code, naf_label)
VALUES ('UNKNOWN', 'Unknown NAF')
ON CONFLICT (naf_code) DO NOTHING;

-- Flat geographic dimension: one row per code_postal with département and région.
-- Allows GROUP BY at postal code, département or région level without extra JOINs.
-- Source: data/20230823-communes-departement-region.csv (INSEE).
-- Latitude/longitude are centroid estimates (one commune per postal code, deduplicated).
CREATE TABLE IF NOT EXISTS gold.dim_geo (
    geo_key          BIGSERIAL PRIMARY KEY,
    code_postal      TEXT NOT NULL UNIQUE,
    code_departement TEXT,
    nom_departement  TEXT,
    code_region      TEXT,
    nom_region       TEXT,
    latitude         NUMERIC(10, 6),
    longitude        NUMERIC(10, 6),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO gold.dim_geo (code_postal, code_departement, nom_departement, code_region, nom_region)
VALUES ('UNKNOWN', 'UNKNOWN', 'Unknown', 'UNKNOWN', 'Unknown')
ON CONFLICT (code_postal) DO NOTHING;

CREATE TABLE IF NOT EXISTS gold.dim_code_rome (
    rome_key BIGSERIAL PRIMARY KEY,
    rome_code TEXT NOT NULL UNIQUE,
    rome_label TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gold.dim_type_contrat (
    contract_key BIGSERIAL PRIMARY KEY,
    contract_type TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gold.dim_experience (
    experience_key BIGSERIAL PRIMARY KEY,
    experience_level TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unknown members to avoid NULL foreign keys
INSERT INTO gold.dim_code_rome (rome_code, rome_label)
VALUES ('UNKNOWN', 'Unknown ROME')
ON CONFLICT (rome_code) DO NOTHING;

INSERT INTO gold.dim_type_contrat (contract_type)
VALUES ('UNKNOWN')
ON CONFLICT (contract_type) DO NOTHING;

INSERT INTO gold.dim_experience (experience_level)
VALUES ('UNKNOWN')
ON CONFLICT (experience_level) DO NOTHING;

-- -----------------------------
-- Minimal fact (1 row per offer)
-- -----------------------------
CREATE TABLE IF NOT EXISTS gold.fact_offre_emploi (
    fact_id BIGSERIAL PRIMARY KEY,
    offer_id TEXT NOT NULL,
    source TEXT NOT NULL,
    snapshot_dt DATE NOT NULL,

    geo_key        BIGINT NOT NULL REFERENCES gold.dim_geo(geo_key),
    naf_key        BIGINT NOT NULL REFERENCES gold.dim_naf(naf_key),
    rome_key       BIGINT NOT NULL REFERENCES gold.dim_code_rome(rome_key),
    contract_key   BIGINT NOT NULL REFERENCES gold.dim_type_contrat(contract_key),
    experience_key BIGINT NOT NULL REFERENCES gold.dim_experience(experience_key),

    status TEXT,
    published_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    unpublished_at TIMESTAMPTZ,

    title TEXT,
    description TEXT,
    job_postal_code TEXT,
    job_city TEXT,
    company_name TEXT,
    company_city TEXT,
    company_postal_code TEXT,
    salary_min NUMERIC(12,2),
    salary_max NUMERIC(12,2),
    salary_periodicity TEXT,
    periodicity TEXT,
    yearly_min NUMERIC(12,2),
    yearly_max NUMERIC(12,2),
    salary_min_computed NUMERIC(12,2),
    salary_max_computed NUMERIC(12,2),

    load_run_id TEXT NOT NULL,
    load_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (offer_id, source)
);

CREATE INDEX IF NOT EXISTS idx_fact_snapshot_dt ON gold.fact_offre_emploi(snapshot_dt);
CREATE INDEX IF NOT EXISTS idx_fact_status ON gold.fact_offre_emploi(status);
CREATE INDEX IF NOT EXISTS idx_fact_source ON gold.fact_offre_emploi(source);

-- -----------------------------
-- Table de suivi des imports incrémentaux
-- -----------------------------
CREATE TABLE IF NOT EXISTS gold.imported_snapshots (
    run_id TEXT NOT NULL PRIMARY KEY,
    snapshot_dt DATE NOT NULL,
    source TEXT NOT NULL,
    import_ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

