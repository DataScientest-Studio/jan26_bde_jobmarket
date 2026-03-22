"""
Tab 3 — Salaires
"""
import streamlit as st
import plotly.graph_objects as go

from utils.queries import (
    load_salaires_distrib,
    load_salaires_par_contrat,
    load_salaires_par_rome,
    load_salaires_par_region,
)
from utils.helpers import kpi_card, base_layout, fmt_euro
from config import GOLD, TEAL, CORAL, PURPLE, DARK_BG, GRID_COL, TEXT_COL


def render(filters_key: str):
    with st.spinner("Chargement…"):
        df_distrib  = load_salaires_distrib(filters_key)
        df_sal_ct   = load_salaires_par_contrat(filters_key)
        df_sal_rome = load_salaires_par_rome(filters_key)
        df_sal_reg  = load_salaires_par_region(filters_key)

    # ── KPIs salaires ──────────────────────────────────────────────────────
    if not df_distrib.empty:
        median_sal = df_distrib["salaire_moyen"].median()
        mean_sal   = df_distrib["salaire_moyen"].mean()
        p25_sal    = df_distrib["salaire_moyen"].quantile(0.25)
        p75_sal    = df_distrib["salaire_moyen"].quantile(0.75)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card("Salaire médian",      fmt_euro(median_sal), "brut annuel",              GOLD),   unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("Salaire moyen",       fmt_euro(mean_sal),   "brut annuel",              TEAL),   unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("1er quartile (P25)",  fmt_euro(p25_sal),    "25% des offres en dessous", CORAL),  unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("3ème quartile (P75)", fmt_euro(p75_sal),    "75% des offres en dessous", PURPLE), unsafe_allow_html=True)

    # ── Distribution ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Distribution des salaires</div>', unsafe_allow_html=True)

    if not df_distrib.empty:
        col_l, col_r = st.columns(2)
        with col_l:
            fig_hist = go.Figure(go.Histogram(
                x=df_distrib["salaire_moyen"],
                nbinsx=50,
                marker=dict(color=GOLD, opacity=0.75, line=dict(color=DARK_BG, width=0.5)),
                hovertemplate="Tranche: %{x:,.0f} €<br>Nb offres: %{y}<extra></extra>",
            ))
            fig_hist.add_vline(
                x=median_sal, line=dict(color=TEAL, dash="dash", width=1.5),
                annotation_text="médiane", annotation_font_color=TEAL,
            )
            fig_hist.update_layout(**base_layout("Distribution des salaires annuels"), height=320)
            fig_hist.update_xaxes(title_text="€ brut / an", tickformat=",.0f")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_r:
            fig_box = go.Figure(go.Box(
                y=df_distrib["salaire_moyen"],
                marker=dict(color=GOLD, size=3),
                line=dict(color=GOLD),
                fillcolor="rgba(212,168,75,0.15)",
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
            marker=dict(color="rgba(74,157,143,0.3)", line=dict(color=TEAL, width=1)),
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
