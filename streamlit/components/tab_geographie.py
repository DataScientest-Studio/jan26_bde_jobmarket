"""
Tab 2 — Géographie
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.queries import load_regions, load_departements, load_top_villes
from utils.helpers import kpi_card, base_layout, fmt_number, fmt_euro
from config import TEAL, PURPLE


def render(filters_key: str):
    with st.spinner("Chargement…"):
        df_reg  = load_regions(filters_key)
        df_dep  = load_departements(filters_key)
        df_vill = load_top_villes(filters_key)

    # ── Par région ─────────────────────────────────────────────────────────
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
            for _, r in df_reg.head(3).iterrows():
                sub = (
                    f"offres · sal. moy. {fmt_euro(r['salaire_moyen'])}"
                    if pd.notna(r["salaire_moyen"])
                    else "offres"
                )
                st.markdown(kpi_card(r["nom_region"], fmt_number(r["nb_offres"]), sub, TEAL), unsafe_allow_html=True)
                st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)

    # ── Top 20 départements ────────────────────────────────────────────────
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

    # ── Tableau départements avec filtre région ────────────────────────────
    st.markdown('<div class="section-title">Détail par département</div>', unsafe_allow_html=True)

    if not df_dep.empty:
        regions_dispo = ["Toutes"] + sorted(df_dep["nom_region"].dropna().unique().tolist())
        sel_reg = st.selectbox("Filtrer par région", regions_dispo, key="sel_reg_tab")

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

    # ── Top 20 villes ──────────────────────────────────────────────────────
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
