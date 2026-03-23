"""
Toutes les fonctions de chargement de données depuis PostgreSQL.
Chaque fonction accepte un paramètre `filters_key` (str) pour invalider
le cache @st.cache_data, et lit les filtres actifs via st.session_state.
"""
import pandas as pd
import streamlit as st
from sqlalchemy import text
from utils.db import get_engine, _sql
from config import DB_TTL

# ── Construction de la clause WHERE dynamique ─────────────────────────────────

def _build_where(filters: dict) -> str:
    clauses = []

    if filters.get("regions"):
        vals = ", ".join(f"'{v.replace(chr(39), chr(39)*2)}'" for v in filters["regions"])
        clauses.append(f"g.nom_region IN ({vals})")

    if filters.get("departements"):
        vals = ", ".join(f"'{v.replace(chr(39), chr(39)*2)}'" for v in filters["departements"])
        clauses.append(f"g.nom_departement IN ({vals})")

    if filters.get("villes"):
        vals = ", ".join(f"'{v.replace(chr(39), chr(39)*2)}'" for v in filters["villes"])
        clauses.append(f"g.nom_commune IN ({vals})")

    if filters.get("contrats"):
        vals = ", ".join(f"'{v.replace(chr(39), chr(39)*2)}'" for v in filters["contrats"])
        clauses.append(f"c.contract_type IN ({vals})")

    if filters.get("secteurs"):
        vals = ", ".join(f"'{v.replace(chr(39), chr(39)*2)}'" for v in filters["secteurs"])
        clauses.append(f"r.rome_label IN ({vals})")

    if filters.get("postes"):
        val = filters["postes"].replace("'", "''")
        clauses.append(f"f.job_title ILIKE '%{val}%'")

    if filters.get("date_debut"):
        clauses.append(f"f.published_at >= '{filters['date_debut']}'")

    if filters.get("date_fin"):
        clauses.append(f"f.published_at <= '{filters['date_fin']} 23:59:59'")

    return ("AND " + " AND ".join(clauses)) if clauses else ""


# ── JOINs conditionnels ───────────────────────────────────────────────────────

def _geo_join(filters: dict) -> str:
    if any(filters.get(k) for k in ("regions", "departements", "villes")):
        return "JOIN gold.dim_geo g ON g.geo_key = f.geo_key"
    return ""

def _contrat_join(filters: dict) -> str:
    if filters.get("contrats"):
        return "JOIN gold.dim_type_contrat c ON c.contract_key = f.contract_key"
    return ""

def _rome_join(filters: dict) -> str:
    if filters.get("secteurs"):
        return "JOIN gold.dim_code_rome r ON r.rome_key = f.rome_key"
    return ""


# ── Options sidebar (régions, départements, contrats) ─────────────────────────

@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_filter_options():
    engine = get_engine()

    regions = _sql(
        "SELECT DISTINCT nom_region FROM gold.dim_geo "
        "WHERE nom_region IS NOT NULL AND code_region != 'UNKNOWN' "
        "ORDER BY nom_region",
        engine,
    )["nom_region"].tolist()

    departements = _sql(
        "SELECT DISTINCT nom_departement, nom_region FROM gold.dim_geo "
        "WHERE nom_departement IS NOT NULL AND code_departement != 'UNKNOWN' "
        "ORDER BY nom_departement",
        engine,
    )

    contrats = _sql(
        "SELECT DISTINCT contract_type FROM gold.dim_type_contrat "
        "WHERE contract_type != 'UNKNOWN' ORDER BY contract_type",
        engine,
    )["contract_type"].tolist()

    return {"regions": regions, "departements": departements, "contrats": contrats}


# ── Recherche dynamique villes ────────────────────────────────────────────────

@st.cache_data(ttl=DB_TTL, show_spinner=False)
def search_villes(prefix: str, regions: tuple = (), departements: tuple = ()) -> list:
    """Retourne les villes commençant par `prefix` (min 2 caractères)."""
    if len(prefix) < 2:
        return []
    engine = get_engine()
    safe = prefix.replace("'", "''")
    clauses = [f"nom_commune ILIKE '{safe}%'", "nom_commune IS NOT NULL"]
    if regions:
        vals = ", ".join(f"'{r.replace(chr(39), chr(39)*2)}'" for r in regions)
        clauses.append(f"nom_region IN ({vals})")
    if departements:
        vals = ", ".join(f"'{d.replace(chr(39), chr(39)*2)}'" for d in departements)
        clauses.append(f"nom_departement IN ({vals})")
    where = " AND ".join(clauses)
    sql = f"SELECT DISTINCT nom_commune FROM gold.dim_geo WHERE {where} ORDER BY nom_commune LIMIT 50"
    return _sql(sql, engine)["nom_commune"].tolist()


# ── Recherche dynamique ROME ──────────────────────────────────────────────────

@st.cache_data(ttl=DB_TTL, show_spinner=False)
def search_rome(query: str) -> list:
    """Recherche par libellé (ILIKE %query%) ou code (ILIKE query%) — min 2 caractères."""
    if len(query) < 2:
        return []
    engine = get_engine()
    safe = query.replace("'", "''")
    sql = f"""
        SELECT DISTINCT rome_code, rome_label
        FROM gold.dim_code_rome
        WHERE rome_code != 'UNKNOWN'
          AND (rome_label ILIKE '%{safe}%' OR rome_code ILIKE '{safe}%')
        ORDER BY rome_label
        LIMIT 50
    """
    return _sql(sql, engine).to_dict("records")


# ── KPIs ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_kpi_global(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT
            COUNT(*)                                            AS total_offres,
            COUNT(DISTINCT f.company_name)                      AS nb_entreprises,
            ROUND(AVG(f.salary_min_computed + f.salary_max_computed) / 2.0, 0)   AS salaire_moyen,
            COUNT(*) FILTER (WHERE f.status = 'published')      AS offres_actives,
            COUNT(*) FILTER (WHERE f.published_at >= NOW() - INTERVAL '7 days')  AS offres_7j,
            COUNT(*) FILTER (WHERE f.published_at >= NOW() - INTERVAL '30 days') AS offres_30j,
            COUNT(*) FILTER (WHERE f.published_at >= NOW() - INTERVAL '90 days') AS offres_90j,
            COUNT(*) FILTER (
                WHERE f.status = 'archived'
                  AND f.unpublished_at >= NOW() - INTERVAL '14 days'
            )                                                   AS offres_pourvues,
            ROUND(AVG(
                EXTRACT(EPOCH FROM (
                    COALESCE(f.unpublished_at, NOW()) - f.published_at
                )) / 86400.0
            ) FILTER (
                WHERE f.published_at IS NOT NULL
                  AND EXTRACT(EPOCH FROM (COALESCE(f.unpublished_at, NOW()) - f.published_at)) > 0
                  AND EXTRACT(EPOCH FROM (COALESCE(f.unpublished_at, NOW()) - f.published_at)) < 86400 * 365
            ), 1)                                               AS duree_moyenne_jours
        FROM gold.fact_offre_emploi f
        {_geo_join(f)} {_contrat_join(f)} {_rome_join(f)}
        WHERE 1=1 {_build_where(f)}
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_contrats(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT c.contract_type, COUNT(*) AS nb
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_type_contrat c ON c.contract_key = f.contract_key
        {_geo_join(f)} {_rome_join(f)}
        WHERE c.contract_type != 'UNKNOWN' {_build_where(f)}
        GROUP BY c.contract_type ORDER BY nb DESC LIMIT 10
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_anciennete(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT e.experience_level, COUNT(*) AS nb
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_experience e ON e.experience_key = f.experience_key
        {_geo_join(f)} {_contrat_join(f)} {_rome_join(f)}
        WHERE e.experience_level != 'UNKNOWN' {_build_where(f)}
        GROUP BY e.experience_level ORDER BY nb DESC
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_offres_par_jour(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT DATE_TRUNC('day', f.published_at)::date AS jour, COUNT(*) AS nb
        FROM gold.fact_offre_emploi f
        {_geo_join(f)} {_contrat_join(f)} {_rome_join(f)}
        WHERE f.published_at >= NOW() - INTERVAL '90 days' {_build_where(f)}
        GROUP BY 1 ORDER BY 1
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_regions(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT
            g.nom_region, g.code_region,
            COUNT(*) AS nb_offres,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0) AS salaire_moyen
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        {_contrat_join(f)} {_rome_join(f)}
        WHERE g.nom_region IS NOT NULL AND g.code_region != 'UNKNOWN' {_build_where(f)}
        GROUP BY g.nom_region, g.code_region ORDER BY nb_offres DESC
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_departements(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT
            g.nom_departement, g.code_departement, g.nom_region,
            COUNT(*) AS nb_offres,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0) AS salaire_moyen
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        {_contrat_join(f)} {_rome_join(f)}
        WHERE g.nom_departement IS NOT NULL AND g.code_departement != 'UNKNOWN' {_build_where(f)}
        GROUP BY g.nom_departement, g.code_departement, g.nom_region ORDER BY nb_offres DESC
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_top_villes(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT g.nom_commune, g.nom_departement, g.nom_region, COUNT(*) AS nb_offres
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        {_contrat_join(f)} {_rome_join(f)}
        WHERE g.nom_commune IS NOT NULL {_build_where(f)}
        GROUP BY g.nom_commune, g.nom_departement, g.nom_region
        ORDER BY nb_offres DESC LIMIT 20
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_salaires_distrib(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT
            f.salary_min_computed, f.salary_max_computed,
            (f.salary_min_computed + f.salary_max_computed) / 2.0 AS salaire_moyen,
            f.source
        FROM gold.fact_offre_emploi f
        {_geo_join(f)} {_contrat_join(f)} {_rome_join(f)}
        WHERE f.salary_min_computed IS NOT NULL
          AND f.salary_max_computed IS NOT NULL
          AND f.salary_min_computed > 0
          AND f.salary_max_computed < 200000
          {_build_where(f)}
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_salaires_par_contrat(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT
            c.contract_type,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0)::numeric, 0) AS salaire_moyen,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
                  (ORDER BY (f.salary_min_computed + f.salary_max_computed) / 2.0)::numeric, 0) AS p25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                  (ORDER BY (f.salary_min_computed + f.salary_max_computed) / 2.0)::numeric, 0) AS p75,
            COUNT(*) AS nb
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_type_contrat c ON c.contract_key = f.contract_key
        {_geo_join(f)} {_rome_join(f)}
        WHERE f.salary_min_computed IS NOT NULL
          AND f.salary_max_computed IS NOT NULL
          AND f.salary_min_computed > 0
          AND c.contract_type != 'UNKNOWN'
          {_build_where(f)}
        GROUP BY c.contract_type
        HAVING COUNT(*) > 10
        ORDER BY salaire_moyen DESC
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_salaires_par_rome(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT
            r.rome_label, r.rome_code,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0) AS salaire_moyen,
            COUNT(*) AS nb_offres
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_code_rome r ON r.rome_key = f.rome_key
        {_geo_join(f)} {_contrat_join(f)}
        WHERE f.salary_min_computed IS NOT NULL
          AND f.salary_max_computed IS NOT NULL
          AND f.salary_min_computed > 0
          AND r.rome_code != 'UNKNOWN'
          {_build_where(f)}
        GROUP BY r.rome_label, r.rome_code
        HAVING COUNT(*) >= 5
        ORDER BY salaire_moyen DESC LIMIT 20
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_salaires_par_region(filters_key: str = ""):
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT
            g.nom_region,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0) AS salaire_moyen,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY (f.salary_min_computed + f.salary_max_computed) / 2.0)::numeric, 0) AS mediane,
            COUNT(*) AS nb_offres
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        {_contrat_join(f)} {_rome_join(f)}
        WHERE f.salary_min_computed IS NOT NULL
          AND f.salary_max_computed IS NOT NULL
          AND f.salary_min_computed > 0
          AND g.nom_region IS NOT NULL
          AND g.code_region != 'UNKNOWN'
          {_build_where(f)}
        GROUP BY g.nom_region
        HAVING COUNT(*) >= 5
        ORDER BY salaire_moyen DESC
    """
    return _sql(sql, engine)


# ── Données carte ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_carte_regions():
    engine = get_engine()
    return pd.read_sql("""
        SELECT
            g.nom_region                                            AS nom,
            g.code_region                                          AS code,
            COUNT(*)                                               AS nb_offres,
            COUNT(DISTINCT f.company_name)                         AS nb_entreprises,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0)    AS salaire_moyen,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY (f.salary_min_computed + f.salary_max_computed) / 2.0)::numeric, 0) AS salaire_mediane,
            COUNT(*) FILTER (WHERE f.published_at >= NOW() - INTERVAL '30 days') AS offres_30j
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        WHERE g.nom_region IS NOT NULL
          AND g.code_region != 'UNKNOWN'
        GROUP BY g.nom_region, g.code_region
        ORDER BY nb_offres DESC
    """, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_carte_departements():
    engine = get_engine()
    return pd.read_sql("""
        SELECT
            g.nom_departement                                      AS nom,
            g.code_departement                                     AS code,
            g.nom_region,
            COUNT(*)                                               AS nb_offres,
            COUNT(DISTINCT f.company_name)                         AS nb_entreprises,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0)    AS salaire_moyen,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY (f.salary_min_computed + f.salary_max_computed) / 2.0)::numeric, 0) AS salaire_mediane,
            COUNT(*) FILTER (WHERE f.published_at >= NOW() - INTERVAL '30 days') AS offres_30j
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        WHERE g.nom_departement IS NOT NULL
          AND g.code_departement != 'UNKNOWN'
        GROUP BY g.nom_departement, g.code_departement, g.nom_region
        ORDER BY nb_offres DESC
    """, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_top_contrats_zone(zone_type: str, zone_nom: str):
    """Top 5 contrats pour une région ou un département donné."""
    engine = get_engine()
    col = "g.nom_region" if zone_type == "region" else "g.nom_departement"
    safe = zone_nom.replace("'", "''")
    return pd.read_sql(f"""
        SELECT c.contract_type, COUNT(*) AS nb
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        JOIN gold.dim_type_contrat c ON c.contract_key = f.contract_key
        WHERE {col} = '{safe}'
          AND c.contract_type != 'UNKNOWN'
        GROUP BY c.contract_type
        ORDER BY nb DESC
        LIMIT 5
    """, engine)


# ── Flux de publication (jour ou semaine) ─────────────────────────────────────

@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_offres_par_periode(granularite: str = "semaine", filters_key: str = ""):
    """
    granularite : 'jour' ou 'semaine'
    Retourne nb offres par jour / semaine sur les 365 derniers jours.
    """
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    trunc = "week" if granularite == "semaine" else "day"
    sql = f"""
        SELECT
            DATE_TRUNC('{trunc}', f.published_at)::date AS periode,
            COUNT(*) AS nb
        FROM gold.fact_offre_emploi f
        {_geo_join(f)} {_contrat_join(f)} {_rome_join(f)}
        WHERE f.published_at >= NOW() - INTERVAL '365 days'
          {_build_where(f)}
        GROUP BY 1
        ORDER BY 1
    """
    return _sql(sql, engine)


# ── Codes NAF par région ──────────────────────────────────────────────────────

@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_naf_par_region(region: str = "", top_n: int = 20, filters_key: str = ""):
    """
    Retourne le top N des codes NAF (naf_code + naf_label) pour une région donnée.
    Si region est vide, retourne le top global.
    """
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    region_clause = ""
    if region and region != "Toutes":
        safe_region = region.replace("'", "''")
        region_clause = f"AND g.nom_region = '{safe_region}'"
    sql = f"""
        SELECT
            n.naf_code,
            n.naf_label,
            COUNT(*)                                               AS nb_offres,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0)   AS salaire_moyen
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_naf n       ON n.naf_key  = f.naf_key
        JOIN gold.dim_geo g       ON g.geo_key  = f.geo_key
        {_contrat_join(f)} {_rome_join(f)}
        WHERE n.naf_code IS NOT NULL
          AND n.naf_code != 'UNKNOWN'
          {region_clause}
          {_build_where(f)}
        GROUP BY n.naf_code, n.naf_label
        ORDER BY nb_offres DESC
        LIMIT {top_n}
    """
    return _sql(sql, engine)


@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_regions_list(filters_key: str = ""):
    """Liste des régions disponibles pour le filtre NAF."""
    engine = get_engine()
    return _sql(
        "SELECT DISTINCT nom_region FROM gold.dim_geo "
        "WHERE nom_region IS NOT NULL AND code_region != 'UNKNOWN' "
        "ORDER BY nom_region",
        engine,
    )["nom_region"].tolist()


# ── Offres par semaine ────────────────────────────────────────────────────────

@st.cache_data(ttl=DB_TTL, show_spinner=False)
def load_offres_par_semaine(filters_key: str = ""):
    """Nombre d'offres publiées par semaine — 26 dernières semaines avec label Sxx."""
    engine = get_engine()
    f = st.session_state.get("active_filters", {})
    sql = f"""
        SELECT
            DATE_TRUNC('week', f.published_at)::date AS semaine,
            TO_CHAR(DATE_TRUNC('week', f.published_at), 'IYYY-IW') AS annee_semaine,
            CONCAT('S', TO_CHAR(DATE_TRUNC('week', f.published_at), 'IW')) AS label_semaine,
            COUNT(*) AS nb
        FROM gold.fact_offre_emploi f
        {_geo_join(f)} {_contrat_join(f)} {_rome_join(f)}
        WHERE f.published_at >= NOW() - INTERVAL '26 weeks'
          {_build_where(f)}
        GROUP BY 1, 2, 3 ORDER BY 1
    """
    return _sql(sql, engine)
