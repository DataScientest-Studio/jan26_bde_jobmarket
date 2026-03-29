import streamlit as st

# ── Page config (doit être le 1er appel Streamlit) ────────────────────────────
st.set_page_config(
    page_title="Job Market · Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports internes ──────────────────────────────────────────────────────────
from components.styles import inject_styles
from components.sidebar import render_sidebar
from components import tab_vue_globale, tab_geographie, tab_salaires, tab_carte, tab_entreprises

# ── Styles CSS ────────────────────────────────────────────────────────────────
inject_styles()

# ── Sidebar — filtres (retourne une clé de cache) ─────────────────────────────
filters_key = render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Marché de l\'Emploi</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Tableau de bord analytique</div>', unsafe_allow_html=True)

# Affichage des filtres actifs sous le titre
filters = st.session_state.get("active_filters", {})
actifs = []
if filters.get("regions"):
    actifs.append(f"{', '.join(filters['regions'])}")
if filters.get("departements"):
    actifs.append(f"{', '.join(filters['departements'])}")
if filters.get("villes"):
    actifs.append(f"{', '.join(filters['villes'])}")
if filters.get("contrats"):
    actifs.append(f"{', '.join(filters['contrats'])}")
if filters.get("secteurs"):
    actifs.append(f"{len(filters['secteurs'])} secteur(s)")
if filters.get("entreprises"):
    actifs.append(f"{len(filters['entreprises'])} entreprise(s)")
if filters.get("postes"):
    actifs.append(f"\"{filters['postes']}\"")
if filters.get("date_debut"):
    debut = filters["date_debut"]
    fin = filters.get("date_fin", "aujourd'hui")
    actifs.append(f"{debut} → {fin}")

if actifs:
    st.markdown(
        '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1rem">'
        + "".join(
            f'<span style="background:#1e2028;border:1px solid #d4a84b33;border-radius:20px;'
            f'padding:0.2rem 0.8rem;font-size:0.78rem;color:#d4a84b">{tag}</span>'
            for tag in actifs
        )
        + "</div>",
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Vue Globale  ",
    "  Géographie  ",
    "  Salaires  ",
    "  Carte  ",
    "  Entreprises ",
])

with tab1:
    tab_vue_globale.render(filters_key)

with tab2:
    tab_geographie.render(filters_key)

with tab3:
    tab_salaires.render(filters_key)

with tab4:
    tab_carte.render(filters_key)

with tab5:
    tab_entreprises.render(filters_key)
