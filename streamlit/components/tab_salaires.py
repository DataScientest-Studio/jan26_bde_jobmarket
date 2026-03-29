"""
Tab 3 — Salaires
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.queries import (
    load_salaires_distrib,
    load_salaires_par_contrat,
    load_salaires_par_rome,
)
from utils.helpers import kpi_card, base_layout, fmt_euro
from config import GOLD, TEAL, CORAL, PURPLE, DARK_BG, GRID_COL, TEXT_COL


def _add_candles(fig: go.Figure, df: pd.DataFrame, y_col: str, xaxis: str = "x2") -> go.Figure:
    has_sal = all(c in df.columns for c in ("salaire_min", "salaire_p25", "salaire_moyen", "salaire_p75", "salaire_max"))
    if not has_sal:
        return fig

    d = df[df["salaire_moyen"].notna()]

    for _, row in d.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["salaire_min"], row["salaire_max"]],
            y=[row[y_col], row[y_col]],
            mode="lines",
            line=dict(color="rgba(212,168,75,0.40)", width=0.8),
            xaxis=xaxis,
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.add_trace(go.Bar(
        x=d["salaire_p75"] - d["salaire_p25"],
        y=d[y_col],
        orientation="h",
        base=d["salaire_p25"],
        marker=dict(color="rgba(212,168,75,0.45)", line=dict(color="#d4a84b", width=0.8)),
        xaxis=xaxis,
        name="Sal. p25–p75",
        customdata=d["salaire_p75"],
        hovertemplate="<b>%{y}</b><br>p25 : %{base:,.0f} €<br>p75 : %{customdata:,.0f} €<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=d["salaire_moyen"],
        y=d[y_col],
        mode="markers",
        marker=dict(color="#d4a84b", size=6, symbol="diamond"),
        xaxis=xaxis,
        name="Sal. moyen",
        hovertemplate="<b>%{y}</b><br>Sal. moyen : %{x:,.0f} €<extra></extra>",
    ))
    return fig


def render(filters_key: str):
    with st.spinner("Chargement…"):
        df_distrib  = load_salaires_distrib(filters_key)
        df_sal_ct   = load_salaires_par_contrat(filters_key)
        df_sal_rome = load_salaires_par_rome(filters_key)

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
            customdata=df_sal_ct["p75"],
            marker=dict(color="rgba(74,157,143,0.3)", line=dict(color=TEAL, width=1)),
            hovertemplate="<b>%{x}</b><br>P25: %{base:,.0f} €<br>P75: %{customdata:,.0f} €<extra></extra>",
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

    # ── Top ROME par salaire ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Top 20 métiers (ROME) par salaire moyen</div>', unsafe_allow_html=True)
    tri_rome = st.radio(
        "Tri (ROME)",
        options=["Nombre d'offres", "Salaire moyen"],
        horizontal=True,
        key="tri_salaires_rome",
    )

    if not df_sal_rome.empty:
        if tri_rome == "Salaire moyen":
            df_sal_rome = df_sal_rome.sort_values(["salaire_moyen", "nb_offres"], ascending=[False, False])
        else:
            df_sal_rome = df_sal_rome.sort_values(["nb_offres", "salaire_moyen"], ascending=[False, False])
        df_sal_rome = df_sal_rome.head(20)

        fig_rome = go.Figure(go.Bar(
            x=df_sal_rome["nb_offres"],
            y=df_sal_rome["rome_label"],
            orientation="h",
            marker=dict(
                color=df_sal_rome["nb_offres"],
                colorscale=[[0, "#423528"], [1, PURPLE]],
                showscale=False,
            ),
            customdata=df_sal_rome[["salaire_moyen", "rome_code"]],
            hovertemplate="<b>%{y}</b><br>%{x:,} offres<br>%{customdata[0]:,.0f} €/an · %{customdata[1]}<extra></extra>",
        ))
        fig_rome = _add_candles(fig_rome, df_sal_rome, y_col="rome_label", xaxis="x2")
        layout_rome = base_layout("Top 20 codes ROME — volume et bougies salariales")
        layout_rome.update(
            height=560,
            barmode="overlay",
            xaxis=dict(
                title="Nb offres",
                gridcolor=layout_rome["xaxis"]["gridcolor"],
                linecolor=layout_rome["xaxis"]["linecolor"],
            ),
            xaxis2=dict(
                title="Salaire annuel (€)",
                overlaying="x",
                side="top",
                showgrid=False,
            ),
        )
        fig_rome.update_layout(**layout_rome)
        fig_rome.update_yaxes(autorange="reversed", tickfont=dict(size=11))
        st.plotly_chart(fig_rome, use_container_width=True)
