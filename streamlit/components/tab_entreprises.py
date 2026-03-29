"""
Tab Entreprises 2 — barres d'offres + bougies salariales horizontales.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import GOLD, PURPLE
from utils.helpers import base_layout
from utils.queries import load_top_entreprises_candles


def render(filters_key: str):
    st.markdown('<div class="section-title">Entreprises et bougies salaires</div>', unsafe_allow_html=True)

    top_n = st.slider("Top N entreprises", min_value=10, max_value=50, value=25, step=5, key="top_n_entreprises_2")

    with st.spinner("Chargement…"):
        df = load_top_entreprises_candles(filters_key=filters_key, limit=top_n)

    if df is None or df.empty:
        st.info("Aucune donnée disponible pour ce graphique.")
        return

    df = df.copy()
    df["entreprise"] = df["entreprise"].fillna("—").astype(str)
    for c in ("nb_offres", "salaire_moyen", "salaire_p25", "salaire_p75", "salaire_min", "salaire_max"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("nb_offres", ascending=False)

    show_unknown = st.checkbox(
        "Inclure les entreprises 'Unknown'",
        value=False,
        key="show_unknown_entreprises",
    )
    if not show_unknown:
        mask = ~df["entreprise"].str.lower().str.strip().isin(["unknown", "—", ""])
        df = df[mask]

    if df.empty:
        st.info("Aucune entreprise identifiée avec les filtres actuels.")
        return

    tri_entreprises = st.selectbox(
        "Trier les entreprises par",
        options=[
            "Volume d'offres décroissant",
            "Volume d'offres croissant",
            "Salaire moyen décroissant",
            "Salaire moyen croissant",
            "Nom (A → Z)",
            "Nom (Z → A)",
        ],
        index=0,
        key="tri_entreprises_tab",
    )

    if tri_entreprises == "Volume d'offres croissant":
        df = df.sort_values(["nb_offres", "entreprise"], ascending=[True, True])
    elif tri_entreprises == "Salaire moyen décroissant":
        df = df.sort_values(["salaire_moyen", "nb_offres"], ascending=[False, False], na_position="last")
    elif tri_entreprises == "Salaire moyen croissant":
        df = df.sort_values(["salaire_moyen", "nb_offres"], ascending=[True, False], na_position="last")
    elif tri_entreprises == "Nom (A → Z)":
        df = df.sort_values("entreprise", ascending=True)
    elif tri_entreprises == "Nom (Z → A)":
        df = df.sort_values("entreprise", ascending=False)
    else:
        df = df.sort_values(["nb_offres", "entreprise"], ascending=[False, True])

    fig = go.Figure()

    # Barres horizontales du volume d'offres
    fig.add_trace(go.Bar(
        x=df["nb_offres"],
        y=df["entreprise"],
        orientation="h",
        marker=dict(
            color=df["nb_offres"],
            colorscale=[[0, "#4d8d5e"], [1, PURPLE]],
            showscale=False,
            ),
        name="Offres",
        hovertemplate="<b>%{y}</b><br>%{x:,} offres<extra></extra>",
    ))

    # "Bougie" = boîte P25-P75 + mèche min-max + point moyenne sur axe X2
    fig.add_trace(go.Bar(
        x=(df["salaire_p75"] - df["salaire_p25"]).clip(lower=0),
        y=df["entreprise"],
        base=df["salaire_p25"],
        orientation="h",
        xaxis="x2",
        marker=dict(color="rgba(212,168,75,0.25)", line=dict(color=GOLD, width=1)),
        name="P25-P75",
        customdata=df["salaire_p75"],
        hovertemplate="<b>%{y}</b><br>P25: %{base:,.0f} €<br>P75: %{customdata:,.0f} €<extra></extra>",
    ))

    for _, r in df.iterrows():
        if pd.notna(r["salaire_min"]) and pd.notna(r["salaire_max"]):
            fig.add_trace(go.Scatter(
                x=[r["salaire_min"], r["salaire_max"]],
                y=[r["entreprise"], r["entreprise"]],
                mode="lines",
                xaxis="x2",
                line=dict(color=GOLD, width=1),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.add_trace(go.Scatter(
        x=df["salaire_moyen"],
        y=df["entreprise"],
        mode="markers",
        xaxis="x2",
        marker=dict(color=GOLD, size=7, symbol="diamond"),
        name="Salaire moyen",
        hovertemplate="<b>%{y}</b><br>Sal. moyen: %{x:,.0f} €<extra></extra>",
    ))

    layout = base_layout("Offres et bougies salariales par entreprise")
    layout.update(
        height=920,
        bargap=0.32,
        barmode="overlay",
        xaxis=dict(
            title="Nb offres",
            gridcolor=layout["xaxis"]["gridcolor"],
            linecolor=layout["xaxis"]["linecolor"],
        ),
        xaxis2=dict(
            title="Salaire (€)",
            overlaying="x",
            side="top",
            showgrid=False,
        ),
    )
    fig.update_layout(**layout)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)