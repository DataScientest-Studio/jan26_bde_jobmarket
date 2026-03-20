import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

from utils.db import query

# Constants
STATUS_UNPUBLISHED = "unpublished"
STATUS_PUBLISHED = "published"

_JOINS = """
    FROM gold.fact_offre_emploi f
    JOIN gold.dim_geo g          ON f.geo_key        = g.geo_key
    JOIN gold.dim_type_contrat c ON f.contract_key    = c.contract_key
    JOIN gold.dim_experience e   ON f.experience_key  = e.experience_key
    JOIN gold.dim_code_rome r    ON f.rome_key        = r.rome_key
    JOIN gold.dim_naf n          ON f.naf_key         = n.naf_key
"""

# ─────────────────────────────────────────────
#  Load environment variables
# ─────────────────────────────────────────────
from src.config.env import load_project_env
load_project_env()  # safe à rappeler (idempotent)


# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Market · Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0f1117;
    color: #e8e6e0;
}

/* Main title */
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #f0ede6;
    letter-spacing: -0.02em;
    margin-bottom: 0.1rem;
}
.main-subtitle {
    font-size: 0.95rem;
    color: #6b6a66;
    margin-bottom: 2rem;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #2a2a30;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #5a5a60;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.8rem 1.6rem;
    border-bottom: 2px solid transparent;
    background: transparent !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #d4a84b !important;
    border-bottom: 2px solid #d4a84b !important;
    background: transparent !important;
}

/* KPI Cards */
.kpi-card {
    background: #16181f;
    border: 1px solid #24262e;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #3a3a44; }
.kpi-accent {
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 12px 0 0 12px;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5a5a64;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    color: #f0ede6;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #4a4a52;
    font-weight: 300;
}

/* Section headers */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #d4c9b0;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #24262e;
}

/* Charts container */
.chart-card {
    background: #16181f;
    border: 1px solid #24262e;
    border-radius: 12px;
    padding: 1.2rem;
}

/* Hide streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_kpi_global():
    return query(f"""
        SELECT
            COUNT(*)                                            AS total_offres,
            COUNT(DISTINCT company_name)                        AS nb_entreprises,
            ROUND(AVG(salary_min_computed + salary_max_computed) / 2.0, 0)       AS salaire_moyen,
            COUNT(*) FILTER (WHERE status = '{STATUS_PUBLISHED}') AS offres_actives,
            COUNT(*) FILTER (WHERE published_at >= NOW() - INTERVAL '7 days')  AS offres_7j,
            COUNT(*) FILTER (WHERE published_at >= NOW() - INTERVAL '30 days') AS offres_30j,
            COUNT(*) FILTER (WHERE published_at >= NOW() - INTERVAL '90 days') AS offres_90j,
            COUNT(*) FILTER (WHERE status = '{STATUS_UNPUBLISHED}' AND unpublished_at >= NOW() - INTERVAL '14 days')    AS offres_pourvues
        FROM gold.fact_offre_emploi
    """)

def get_map_data(filter_name):

    if(filter_name == "avg_salary"):
        where= """WHERE salary_min_computed IS NOT NULL
            AND salary_max_computed IS NOT NULL
            AND salary_min_computed > 0
            AND salary_max_computed < 200000"""
    else:
        where= """WHERE salary_min_computed IS NOT NULL
            AND salary_max_computed IS NOT NULL
            AND salary_min_computed > 0
            AND salary_max_computed < 200000"""

    return query(f"""
        SELECT
            g.code_departement,
            g.nom_departement,
            g.code_region,
            g.nom_region,
            AVG(g.latitude::float)                                                          AS lat,
            AVG(g.longitude::float)                                                         AS lon,
            COUNT(f.fact_id)                                                                AS nb_offres,
            ROUND(AVG(f.salary_min_computed) FILTER (WHERE f.salary_min_computed IS NOT NULL))::bigint AS avg_salary,
            ROUND(100.0 * COUNT(*) FILTER (WHERE c.contract_type = 'CDI')
                  / NULLIF(COUNT(*), 0), 1)                                                 AS pct_cdi,
            ROUND(100.0 * COUNT(*) FILTER (WHERE e.experience_level IN ('3-5 ans','5-10 ans','10+ ans'))
                  / NULLIF(COUNT(*), 0), 1)                                                 AS pct_senior,
            COUNT(DISTINCT r.rome_code)                                                     AS nb_rome_codes
        {_JOINS}
        {where}
        GROUP BY g.code_departement, g.nom_departement, g.code_region, g.nom_region
        HAVING COUNT(f.fact_id) > 0
        ORDER BY nb_offres DESC
    """)


def get_map(filter_name, col_indicator_size="nb_offres", 
            col_color_value="avg_salary", 
            color_bar_title="Salaire moyen (€/an)", 
            colorscale="oranges"):

    df = get_map_data(filter_name)

    if df.empty:
        st.warning("Aucune donnée.")
        st.stop()

    # Nettoyage : exclure départements DOM-TOM pour la vue principale
    # (lat/lon hors métropole fausseraient le zoom auto)
    df_metro = df[df["lat"].between(41, 52) & df["lon"].between(-5, 10)].copy()
    # On peut choisir d'afficher les DOM-TOM dans une carte séparée ou un tableau en bas de page
    df_dom   = df[~df.index.isin(df_metro.index)].copy()

    # Formater les valeurs pour le hover
    def fmt_val(v):
        if pd.isna(v):
            return "N/A"
        try:
            return fmt_number(v)
        except Exception:
            return str(v)

    df_metro["_indicateur_fmt"] = df_metro[col_color_value].apply(fmt_val)

    # ---------------------------------------------------------------------------
    # Carte principale (métropole)
    # ---------------------------------------------------------------------------
    fig = px.scatter_mapbox(
        df_metro,
        lat="lat",
        lon="lon",
        # Taille des bulles selon le nombre d'offres (ou autre indicateur choisi)
        size=col_indicator_size,
        size_max=60,
        # Couleur d'interpolation selon la valeur de l'indicateur choisi (ex: salaire moyen)
        color=col_color_value,
        color_continuous_scale=colorscale,
        hover_name="nom_departement",
        # Données du hover (code département, région, nombre d'offres, indicateur formaté...)
        hover_data={
            "code_departement": True,
            "nom_region": True,
            "nb_offres": ":,",
            "pct_cdi": ":.1f",
            "pct_senior": ":.1f",
            "nb_rome_codes": ":,",
            "lat": False,
            "lon": False,
            col_color_value: False,
        },
        # Style et zoom auto centré sur la France métropolitaine
        mapbox_style="open-street-map",
        # Zoom auto centré sur la France métropolitaine
        zoom=4.8,
        # Centrer sur la France métropolitaine
        center={"lat": 46.6, "lon": 2.5},
        labels={
            "code_departement": "Dép.",
            "nom_region": "Région",
            "nb_offres": "Offres",
            "avg_salary": "Salaire moy. (€/an)",
            "pct_cdi": "% CDI",
            "pct_senior": "% Senior",
            "nb_rome_codes": "Nb métiers",
        },
    )
   
    fig.update_layout(
        height=600,
        margin=dict(t=0, b=0, l=0, r=0),
        coloraxis_colorbar=dict(title=color_bar_title, thickness=15, len=0.6),
    )
    return fig
    #st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_contrats():
    return query(f"""
        SELECT
            c.contract_type,
            COUNT(*) AS nb
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_type_contrat c ON c.contract_key = f.contract_key
        WHERE c.contract_type != 'UNKNOWN'
        GROUP BY c.contract_type
        ORDER BY nb DESC
        LIMIT 10
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_anciennete():
    return query(f"""
        SELECT
            e.experience_level,
            COUNT(*) AS nb
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_experience e ON e.experience_key = f.experience_key
        WHERE e.experience_level != 'UNKNOWN'
        GROUP BY e.experience_level
        ORDER BY nb DESC
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_offres_par_jour():
    return query(f"""
        SELECT
            DATE_TRUNC('day', published_at)::date AS jour,
            COUNT(*) AS nb
        FROM gold.fact_offre_emploi
        WHERE published_at >= NOW() - INTERVAL '90 days'
        GROUP BY 1
        ORDER BY 1
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_regions():
    return query(f"""
        SELECT
            g.nom_region,
            g.code_region,
            COUNT(*)                                        AS nb_offres,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0) AS salaire_moyen
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        WHERE g.nom_region IS NOT NULL
          AND g.code_region != 'UNKNOWN'
        GROUP BY g.nom_region, g.code_region
        ORDER BY nb_offres DESC
    """ )

@st.cache_data(ttl=300, show_spinner=False)
def load_departements():
    return query(f"""
        SELECT
            g.nom_departement,
            g.code_departement,
            g.nom_region,
            COUNT(*)                                        AS nb_offres,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0) AS salaire_moyen
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        WHERE g.nom_departement IS NOT NULL
          AND g.code_departement != 'UNKNOWN'
        GROUP BY g.nom_departement, g.code_departement, g.nom_region
        ORDER BY nb_offres DESC
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_top_villes():
    return query(f"""
        SELECT
            g.nom_commune,
            g.nom_departement,
            g.nom_region,
            COUNT(*) AS nb_offres
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        WHERE g.nom_commune IS NOT NULL
        GROUP BY g.nom_commune, g.nom_departement, g.nom_region
        ORDER BY nb_offres DESC
        LIMIT 20
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_salaires_distrib():
    return query(f"""
        SELECT
            salary_min_computed,
            salary_max_computed,
            (salary_min_computed + salary_max_computed) / 2.0 AS salaire_moyen,
            source
        FROM gold.fact_offre_emploi
        WHERE salary_min_computed IS NOT NULL
          AND salary_max_computed IS NOT NULL
          AND salary_min_computed > 0
          AND salary_max_computed < 200000
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_salaires_par_contrat():
    return query(f"""
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
        WHERE f.salary_min_computed IS NOT NULL
          AND f.salary_max_computed IS NOT NULL
          AND f.salary_min_computed > 0
          AND c.contract_type != 'UNKNOWN'
        GROUP BY c.contract_type
        HAVING COUNT(*) > 10
        ORDER BY salaire_moyen DESC
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_salaires_par_rome():
    return query(f"""
        SELECT
            r.rome_label,
            r.rome_code,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0) AS salaire_moyen,
            COUNT(*) AS nb_offres
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_code_rome r ON r.rome_key = f.rome_key
        WHERE f.salary_min_computed IS NOT NULL
          AND f.salary_max_computed IS NOT NULL
          AND f.salary_min_computed > 0
          AND r.rome_code != 'UNKNOWN'
        GROUP BY r.rome_label, r.rome_code
        HAVING COUNT(*) >= 5
        ORDER BY salaire_moyen DESC
        LIMIT 20
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_salaires_par_region():
    return query(f"""
        SELECT
            g.nom_region,
            ROUND(AVG((f.salary_min_computed + f.salary_max_computed) / 2.0), 0) AS salaire_moyen,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY (f.salary_min_computed + f.salary_max_computed) / 2.0)::numeric, 0) AS mediane,
            COUNT(*) AS nb_offres
        FROM gold.fact_offre_emploi f
        JOIN gold.dim_geo g ON g.geo_key = f.geo_key
        WHERE f.salary_min_computed IS NOT NULL
          AND f.salary_max_computed IS NOT NULL
          AND f.salary_min_computed > 0
          AND g.nom_region IS NOT NULL
          AND g.code_region != 'UNKNOWN'
        GROUP BY g.nom_region
        HAVING COUNT(*) >= 5
        ORDER BY salaire_moyen DESC
    """)

# ── Plotly theme ──────────────────────────────────────────────────────────────
DARK_BG   = "#16181f"
GRID_COL  = "#24262e"
TEXT_COL  = "#9a9890"
GOLD      = "#d4a84b"
GOLD2     = "#e8c97a"
TEAL      = "#30b439"
CORAL     = "#c96a4a"
PURPLE    = "#7a6aad"

def base_layout(title=""):
    return dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(family="DM Sans", color=TEXT_COL, size=12),
        title=dict(text=title, font=dict(color="#d4c9b0", size=14, family="DM Serif Display"), x=0.02),
        margin=dict(l=16, r=16, t=40 if title else 16, b=16),
        xaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, tickcolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, tickcolor=GRID_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COL),
        hoverlabel=dict(bgcolor="#1e2028", bordercolor=GRID_COL, font_color="#e8e6e0"),
    )

def fmt_number(n):
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)

def kpi_card(label, value, sub="", color=GOLD):
    return f"""
    <div class="kpi-card">
        <div class="kpi-accent" style="background:{color}"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Marché de l\'Emploi</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Tableau de bord analytique</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  Vue Globale  ",
    "  Géographie  ",
    "  Salaires  ",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Vue Globale
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.spinner("Chargement..."):
        kpi   = load_kpi_global()
        df_ct = load_contrats()
        df_an = load_anciennete()
        df_jj = load_offres_par_jour()

    row = kpi.iloc[0]

    # ── KPI row 1 ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Total offres", fmt_number(row.total_offres), "toutes sources", GOLD), unsafe_allow_html=True)
    with c2:
        sal = f"{int(row.salaire_moyen):,} €".replace(",", " ") if pd.notna(row.salaire_moyen) else "—"
        st.markdown(kpi_card("Salaire moyen", sal, "brut annuel estimé", TEAL), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Entreprises", fmt_number(row.nb_entreprises), "recruteurs distincts", CORAL), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Offres actives", fmt_number(row.offres_actives), "statut = active", PURPLE), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── KPI row 2 — ancienneté publication ───────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Offres < 7 jours", fmt_number(row.offres_7j), "publiées cette semaine", GOLD2), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Offres < 30 jours", fmt_number(row.offres_30j), "publiées ce mois", GOLD2), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Offres < 90 jours", fmt_number(row.offres_90j), "publiées ce trimestre", GOLD2), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Offres pourvues:", 
                             fmt_number(row.offres_pourvues),
                             "dans les 14 derniers jours", 
                             GOLD2), unsafe_allow_html=True )

    st.markdown('<div class="section-title">Flux de publication</div>', unsafe_allow_html=True)

    # ── Timeline ──────────────────────────────────────────────────────────
    if not df_jj.empty:
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=df_jj["jour"], y=df_jj["nb"],
            mode="lines",
            fill="tozeroy",
            fillcolor=f"rgba(212,168,75,0.08)",
            line=dict(color=GOLD, width=1.5),
            name="Offres publiées",
            hovertemplate="<b>%{x}</b><br>%{y} offres<extra></extra>",
        ))
        fig_time.update_layout(**base_layout("Offres publiées par jour — 90 derniers jours"), height=240)
        st.plotly_chart(fig_time, use_container_width=True)

    # ── Contrats + Ancienneté ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Répartition des offres</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        if not df_ct.empty:
            fig_ct = go.Figure(go.Bar(
                x=df_ct["nb"],
                y=df_ct["contract_type"],
                orientation="h",
                marker=dict(
                    color=df_ct["nb"],
                    colorscale=[[0, "#2a2410"], [1, GOLD]],
                    showscale=False,
                ),
                hovertemplate="<b>%{y}</b><br>%{x:,} offres<extra></extra>",
            ))
            fig_ct.update_layout(**base_layout("Par type de contrat"), height=320)
            fig_ct.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_ct, use_container_width=True)

with col_r:
    if not df_an.empty:
        fig_an = go.Figure(go.Bar(
            x=df_an["nb"],
            y=df_an["experience_level"],
            orientation="h",
            marker=dict(
                color=df_an["nb"],
                colorscale=[[0, "#2a1a10"], [1, CORAL]],
                showscale=False,
            ),
            hovertemplate="<b>%{y}</b><br>%{x:,} offres<extra></extra>",
        ))
        fig_an.update_layout(**base_layout("Par niveau d'expérience"), height=320)
        fig_an.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_an, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Géographie
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    with st.spinner("Chargement..."):
        df_reg  = load_regions()
        df_dep  = load_departements()
        df_vill = load_top_villes()

    st.markdown('<div class="section-title">Par région</div>', unsafe_allow_html=True)

    if not df_reg.empty:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            fig_reg = go.Figure(go.Bar(
                x=df_reg["nb_offres"],
                y=df_reg["nom_region"],
                orientation="h",
                marker=dict(
                    color=df_reg["nb_offres"],
                    colorscale=[[0, "#1a2a28"], [1, TEAL]],
                    showscale=False,
                ),
                hovertemplate="<b>%{y}</b><br>%{x:,} offres<extra></extra>",
            ))
            fig_reg.update_layout(**base_layout("Nombre d'offres par région"), height=460)
            fig_reg.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_reg, use_container_width=True)

        with col_r:
            st.markdown('<div style="margin-top:2.5rem"></div>', unsafe_allow_html=True)
            top3 = df_reg.head(3)
            for _, r in top3.iterrows():
                st.markdown(kpi_card(
                    r["nom_region"],
                    fmt_number(r["nb_offres"]),
                    f"offres · sal. moy. {int(r['salaire_moyen']):,} €".replace(",", " ") if pd.notna(r["salaire_moyen"]) else "offres",
                    TEAL
                ), unsafe_allow_html=True)
                st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top 20 départements</div>', unsafe_allow_html=True)

    if not df_dep.empty:
        top20 = df_dep.head(20)
        fig_dep = go.Figure()
        fig_dep.add_trace(go.Bar(
            name="Offres",
            x=top20["nom_departement"],
            y=top20["nb_offres"],
            marker=dict(color=TEAL, opacity=0.85),
            hovertemplate="<b>%{x}</b><br>%{y:,} offres<extra></extra>",
        ))
        fig_dep.update_layout(**base_layout("Nombre d'offres par département (top 20)"), height=340, bargap=0.3)
        fig_dep.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_dep, use_container_width=True)

    # Tableau départements complet avec filtre région
    st.markdown('<div class="section-title">Détail par département</div>', unsafe_allow_html=True)
    if not df_dep.empty:
        regions_dispo = ["Toutes"] + sorted(df_dep["nom_region"].dropna().unique().tolist())
        sel_reg = st.selectbox("Filtrer par région", regions_dispo, key="sel_reg")

        df_dep_f = df_dep if sel_reg == "Toutes" else df_dep[df_dep["nom_region"] == sel_reg]

        st.dataframe(
            df_dep_f[["nom_departement", "code_departement", "nom_region", "nb_offres", "salaire_moyen"]]
            .rename(columns={
                "nom_departement":  "Département",
                "code_departement": "Code",
                "nom_region":       "Région",
                "nb_offres":        "Nb offres",
                "salaire_moyen":    "Sal. moyen (€)",
            })
            .reset_index(drop=True),
            use_container_width=True,
            height=340,
        )

    st.markdown('<div class="section-title">Top 20 villes</div>', unsafe_allow_html=True)
    if not df_vill.empty:
        fig_v = go.Figure(go.Bar(
            x=df_vill["nb_offres"],
            y=df_vill["nom_commune"],
            orientation="h",
            marker=dict(
                color=df_vill["nb_offres"],
                colorscale=[[0, "#1a1a28"], [1, PURPLE]],
                showscale=False,
            ),
            hovertemplate="<b>%{y}</b><br>%{x:,} offres<extra></extra>",
        ))
        fig_v.update_layout(**base_layout("Top 20 villes par volume d'offres"), height=440)
        fig_v.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_v, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Salaires
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    with st.spinner("Chargement..."):
        df_distrib  = load_salaires_distrib()
        df_sal_ct   = load_salaires_par_contrat()
        df_sal_rome = load_salaires_par_rome()
        df_sal_reg  = load_salaires_par_region()

    # ── KPIs salaires ─────────────────────────────────────────────────────
    if not df_distrib.empty:
        median_sal = df_distrib["salaire_moyen"].median()
        mean_sal   = df_distrib["salaire_moyen"].mean()
        p75_sal    = df_distrib["salaire_moyen"].quantile(0.75)
        p25_sal    = df_distrib["salaire_moyen"].quantile(0.25)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card("Salaire médian", f"{int(median_sal):,} €".replace(",", " "), "brut annuel", GOLD), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("Salaire moyen", f"{int(mean_sal):,} €".replace(",", " "), "brut annuel", TEAL), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("1er quartile (P25)", f"{int(p25_sal):,} €".replace(",", " "), "25% des offres en dessous", CORAL), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("3ème quartile (P75)", f"{int(p75_sal):,} €".replace(",", " "), "75% des offres en dessous", PURPLE), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribution des salaires</div>', unsafe_allow_html=True)

    if not df_distrib.empty:
        col_l, col_r = st.columns(2)
        with col_l:
            fig_map= get_map(filter_name= "avg_salary", 
                             col_indicator_size="nb_offres", 
                            col_color_value="avg_salary",
                            color_bar_title="Salaire moyen (€/an)", 
                            colorscale="oranges")

            st.plotly_chart(fig_map, use_container_width=True)

            fig_hist = go.Figure(go.Histogram(
                x=df_distrib["salaire_moyen"],
                nbinsx=50,
                marker=dict(color=GOLD, opacity=0.75, line=dict(color=DARK_BG, width=0.5)),
                hovertemplate="Tranche: %{x:,.0f} €<br>Nb offres: %{y}<extra></extra>",
            ))
            fig_hist.add_vline(x=median_sal, line=dict(color=TEAL, dash="dash", width=1.5),
                               annotation_text="médiane", annotation_font_color=TEAL)
            fig_hist.update_layout(**base_layout("Distribution des salaires annuels"), height=320)
            fig_hist.update_xaxes(title_text="€ brut / an", tickformat=",.0f")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_r:
            fig_box = go.Figure(go.Box(
                y=df_distrib["salaire_moyen"],
                marker=dict(color=GOLD, size=3),
                line=dict(color=GOLD),
                fillcolor=f"rgba(212,168,75,0.15)",
                boxmean=True,
                hovertemplate="%{y:,.0f} €<extra></extra>",
                name="Salaires",
            ))
            fig_box.update_layout(**base_layout("Box plot — dispersion"), height=320)
            fig_box.update_yaxes(title_text="€ brut / an", tickformat=",.0f")
            st.plotly_chart(fig_box, use_container_width=True)

    # ── Salaires par contrat ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Salaires par type de contrat</div>', unsafe_allow_html=True)

    if not df_sal_ct.empty:
        fig_ct_sal = go.Figure()
        fig_ct_sal.add_trace(go.Bar(
            name="P25–P75",
            x=df_sal_ct["contract_type"],
            y=df_sal_ct["p75"] - df_sal_ct["p25"],
            base=df_sal_ct["p25"],
            marker=dict(color=f"rgba(74,157,143,0.3)", line=dict(color=TEAL, width=1)),
            hovertemplate="<b>%{x}</b><br>P25: %{base:,.0f} €<br>P75: %{top:,.0f} €<extra></extra>",
        ))
        fig_ct_sal.add_trace(go.Scatter(
            name="Moyenne",
            x=df_sal_ct["contract_type"],
            y=df_sal_ct["salaire_moyen"],
            mode="markers",
            marker=dict(color=GOLD, size=10, symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>Moy: %{y:,.0f} €<extra></extra>",
        ))
        fig_ct_sal.update_layout(**base_layout("Fourchette salariale P25–P75 par contrat"), height=320, bargap=0.4)
        fig_ct_sal.update_yaxes(tickformat=",.0f", title_text="€ / an")
        st.plotly_chart(fig_ct_sal, use_container_width=True)

    # ── Salaires par région ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Salaires par région</div>', unsafe_allow_html=True)

    if not df_sal_reg.empty:
        fig_reg_sal = go.Figure()
        fig_reg_sal.add_trace(go.Bar(
            name="Salaire moyen",
            x=df_sal_reg["salaire_moyen"],
            y=df_sal_reg["nom_region"],
            orientation="h",
            marker=dict(
                color=df_sal_reg["salaire_moyen"],
                colorscale=[[0, "#1a2010"], [0.5, TEAL], [1, GOLD]],
                showscale=True,
                colorbar=dict(
                    tickformat=",.0f",
                    ticksuffix=" €",
                    bgcolor=DARK_BG,
                    tickfont=dict(color=TEXT_COL),
                    outlinecolor=GRID_COL,
                ),
            ),
            hovertemplate="<b>%{y}</b><br>Moy: %{x:,.0f} €<br>Médiane: %{customdata:,.0f} €<extra></extra>",
            customdata=df_sal_reg["mediane"],
        ))
        fig_reg_sal.update_layout(**base_layout("Salaire moyen annuel par région"), height=480)
        fig_reg_sal.update_yaxes(autorange="reversed")
        fig_reg_sal.update_xaxes(tickformat=",.0f", title_text="€ brut / an")
        st.plotly_chart(fig_reg_sal, use_container_width=True)

    # ── Top ROME par salaire ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Top 20 métiers (ROME) par salaire moyen</div>', unsafe_allow_html=True)

    if not df_sal_rome.empty:
        fig_rome = go.Figure(go.Bar(
            x=df_sal_rome["salaire_moyen"],
            y=df_sal_rome["rome_label"],
            orientation="h",
            marker=dict(
                color=df_sal_rome["salaire_moyen"],
                colorscale=[[0, "#1a1510"], [1, CORAL]],
                showscale=False,
            ),
            customdata=df_sal_rome[["nb_offres", "rome_code"]],
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} €/an<br>%{customdata[0]} offres · %{customdata[1]}<extra></extra>",
        ))
        fig_rome.update_layout(**base_layout("Top 20 codes ROME — salaire moyen annuel"), height=560)
        fig_rome.update_yaxes(autorange="reversed", tickfont=dict(size=11))
        fig_rome.update_xaxes(tickformat=",.0f", title_text="€ brut / an")
        st.plotly_chart(fig_rome, use_container_width=True)