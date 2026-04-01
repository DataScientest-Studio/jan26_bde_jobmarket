"""
Injection du CSS global de l'application.
"""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Background */
.stApp { background: #0f1117; color: #e8e6e0; }

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
    height: 118px;
    box-sizing: border-box;
    overflow: visible;
    transition: border-color 0.2s;
    display: flex;
    flex-direction: column;
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
    margin-bottom: 0.55rem;
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
    color: #6b6a66;
    font-weight: 300;
    margin-top: auto;
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

/* Hide streamlit defaults */
/* Cacher uniquement le contenu indésirable, PAS le header entier */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Ne pas masquer la toolbar : elle contient aussi le contrôle de sidebar (chevron). */
header [data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>

"""


def inject_styles():
    st.markdown(CSS, unsafe_allow_html=True)


