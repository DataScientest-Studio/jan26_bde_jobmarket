"""
Tab 2 — Géographie
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import re

from utils.queries import load_regions, load_departements, load_top_villes
from utils.helpers import kpi_card, base_layout, fmt_number, fmt_euro
from config import TEAL, PURPLE


def _add_candles(fig: go.Figure, df: pd.DataFrame, y_col: str, xaxis: str = "x2") -> go.Figure:
    """
    Ajoute les bougies salariales (min / p25 / moy / p75 / max) sur l'axe xaxis.
    Utilise des barres d'erreur + marqueur central pour imiter une bougie horizontale.
    """
    has_sal = all(c in df.columns for c in ("salaire_min", "salaire_p25", "salaire_moyen", "salaire_p75", "salaire_max"))
    if not has_sal:
        return fig

    mask = df["salaire_moyen"].notna()
    d = df[mask]

    # Mèche complète (min → max)
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

    # Corps de la bougie (p25 → p75)
    fig.add_trace(go.Bar(
        x=d["salaire_p75"] - d["salaire_p25"],
        y=d[y_col],
        orientation="h",
        base=d["salaire_p25"],
        marker=dict(color="rgba(212,168,75,0.45)", line=dict(color="#d4a84b", width=0.8)),
        xaxis=xaxis,
        name="Sal. p25–p75",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "p25 : %{base:,.0f} €<br>"
            "p75 : %{x:,.0f} €<extra></extra>"
        ),
    ))

    # Marqueur médiane / moyenne
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
        df_reg  = load_regions(filters_key)
        df_dep  = load_departements(filters_key)
        df_vill = load_top_villes(filters_key)

    # ── Par région ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Par région</div>', unsafe_allow_html=True)
    tri_reg = st.radio(
        "Tri (régions)",
        options=["Nombre d'offres", "Salaire moyen"],
        horizontal=True,
        key="tri_geo_regions",
    )

    if not df_reg.empty:
        if tri_reg == "Salaire moyen":
            df_reg = df_reg.sort_values(["salaire_moyen", "nb_offres"], ascending=[False, False], na_position="last")
        else:
            df_reg = df_reg.sort_values(["nb_offres", "salaire_moyen"], ascending=[False, False], na_position="last")

        fig_reg = go.Figure()

        # Barres volume d'offres (axe x principal)
        fig_reg.add_trace(go.Bar(
            x=df_reg["nb_offres"],
            y=df_reg["nom_region"],
            orientation="h",
            marker=dict(
                color=df_reg["nb_offres"],
                colorscale=[[0, "#4d8d5e"], [1, PURPLE]],
                showscale=False,
            ),
            name="Offres",
            hovertemplate="<b>%{y}</b><br>%{x:,} offres<extra></extra>",
        ))

        # Bougies salariales (axe x2)
        fig_reg = _add_candles(fig_reg, df_reg, y_col="nom_region", xaxis="x2")

        layout_reg = base_layout("Nombre d'offres et distribution salariale par région")
        layout_reg.update(
            height=520,
            xaxis=dict(
                title="Nb offres",
                gridcolor=layout_reg["xaxis"]["gridcolor"],
                linecolor=layout_reg["xaxis"]["linecolor"],
            ),
            xaxis2=dict(
                title="Salaire annuel (€)",
                overlaying="x",
                side="top",
                showgrid=False,
            ),
            barmode="overlay",
        )
        fig_reg.update_layout(**layout_reg)
        fig_reg.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_reg, use_container_width=True)

    # ── Top 20 départements ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Top 20 départements</div>', unsafe_allow_html=True)
    tri_dep = st.radio(
        "Tri (départements)",
        options=["Nombre d'offres", "Salaire moyen"],
        horizontal=True,
        key="tri_geo_departements",
    )

    if not df_dep.empty:
        if tri_dep == "Salaire moyen":
            df_dep = df_dep.sort_values(["salaire_moyen", "nb_offres"], ascending=[False, False], na_position="last")
        else:
            df_dep = df_dep.sort_values(["nb_offres", "salaire_moyen"], ascending=[False, False], na_position="last")
        top20 = df_dep.head(20)
        fig_dep = go.Figure()

        # Barres volume
        fig_dep.add_trace(go.Bar(
            name="Offres",
            x=top20["nb_offres"],
            y=top20["nom_departement"],
            orientation="h",
            marker=dict(
                color=top20["nb_offres"],
                colorscale=[[0, "#4d8d5e"], [1, PURPLE]],
                showscale=False,
            ),
            hovertemplate="<b>%{y}</b><br>%{x:,} offres<extra></extra>",
        ))

        # Bougies salariales
        fig_dep = _add_candles(fig_dep, top20, y_col="nom_departement", xaxis="x2")

        layout_dep = base_layout("Offres et distribution salariale par département (top 20)")
        layout_dep.update(
            height=520,
            bargap=0.3,
            xaxis=dict(
                title="Nb offres",
                gridcolor=layout_dep["xaxis"]["gridcolor"],
                linecolor=layout_dep["xaxis"]["linecolor"],
            ),
            xaxis2=dict(
                title="Salaire annuel (€)",
                overlaying="x",
                side="top",
                showgrid=False,
            ),
            barmode="overlay",
        )
        fig_dep.update_layout(**layout_dep)
        fig_dep.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_dep, use_container_width=True)

    # ── Top 20 villes ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Top 20 villes</div>', unsafe_allow_html=True)
    tri_villes = st.radio(
        "Tri (villes)",
        options=["Nombre d'offres", "Salaire moyen"],
        horizontal=True,
        key="tri_geo_villes",
    )

    if not df_vill.empty:
        show_unknown = st.checkbox(
            "Inclure les villes 'Unknown'",
            value=False,
            key="show_unknown_cities_tab",
        )
        if not show_unknown:
            mask_known = df_vill["nom_commune"].fillna("").str.lower().ne("unknown")
            df_vill = df_vill[mask_known]

        group_arrond = st.checkbox(
            "Regrouper les arrondissements (Paris/Lyon/Marseille)",
            value=False,
            key="group_arrond_cities_tab",
        )

        if group_arrond:
            def _arrond_group_key(city: object) -> str:
                s = str(city).strip()
                if not s:
                    return s
                su = s.upper()
                for base in ("PARIS", "LYON", "MARSEILLE"):
                    if su.startswith(base):
                        if re.match(rf"^{base}\s*\d{{1,2}}(?:\s*(?:ER|E|ÈME|EME))?$", su):
                            return base
                return s

            df_vill = df_vill.copy()
            df_vill["ville_groupe"] = df_vill["nom_commune"].apply(_arrond_group_key)

            grouped_rows = []
            sal_cols = ["salaire_moyen", "salaire_p25", "salaire_p75", "salaire_min", "salaire_max"]
            for (nom_region, nom_departement, ville_groupe), g in df_vill.groupby(
                ["nom_region", "nom_departement", "ville_groupe"], dropna=False
            ):
                nb_sum = int(g["nb_offres"].sum())
                row = {
                    "nom_region": nom_region,
                    "nom_departement": nom_departement,
                    "nom_commune": ville_groupe,
                    "nb_offres": nb_sum,
                }
                # Moyenne pondérée pour salaire_moyen, min/max directs, p25/p75 approchés
                for col in sal_cols:
                    if col not in g.columns:
                        row[col] = None
                        continue
                    mask_sal = g[col].notna()
                    if not mask_sal.any():
                        row[col] = None
                    elif col == "salaire_min":
                        row[col] = float(g.loc[mask_sal, col].min())
                    elif col == "salaire_max":
                        row[col] = float(g.loc[mask_sal, col].max())
                    else:
                        # Moyenne pondérée par nb_offres pour moyen/p25/p75
                        row[col] = float(
                            (g.loc[mask_sal, col] * g.loc[mask_sal, "nb_offres"]).sum()
                            / g.loc[mask_sal, "nb_offres"].sum()
                        )
                grouped_rows.append(row)

            df_vill = pd.DataFrame(grouped_rows)

        if tri_villes == "Salaire moyen":
            df_vill = df_vill.sort_values(["salaire_moyen", "nb_offres"], ascending=[False, False], na_position="last")
        else:
            df_vill = df_vill.sort_values(["nb_offres", "salaire_moyen"], ascending=[False, False], na_position="last")
        top20_v = df_vill.head(20)

        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(
            x=top20_v["nb_offres"],
            y=top20_v["nom_commune"],
            orientation="h",
            marker=dict(
                color=top20_v["nb_offres"],
                colorscale=[[0, "#4d8d5e"], [1, PURPLE]],
                showscale=False,
            ),
            name="Offres",
            hovertemplate="<b>%{y}</b><br>%{x:,} offres<extra></extra>",
        ))

        # Bougies salariales villes (p25/p75/min/max disponibles)
        fig_v = _add_candles(fig_v, top20_v, y_col="nom_commune", xaxis="x2")

        layout_v = base_layout("Top 20 villes – offres et salaire moyen")
        layout_v.update(
            height=460,
            xaxis=dict(
                title="Nb offres",
                gridcolor=layout_v["xaxis"]["gridcolor"],
                linecolor=layout_v["xaxis"]["linecolor"],
            ),
            xaxis2=dict(
                title="Salaire annuel (€)",
                overlaying="x",
                side="top",
                showgrid=False,
            ),
        )
        fig_v.update_layout(**layout_v)
        fig_v.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_v, use_container_width=True)