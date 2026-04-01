"""
Tab 1 — Vue Globale
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.queries import (
    load_kpi_global,
    load_contrats_salaire_stats,
    load_anciennete_salaire_stats,
    load_offres_par_jour,
    load_offres_par_semaine,
    load_naf_par_region_salaire_stats,
    load_regions_list,
)
from utils.helpers import kpi_card, base_layout, fmt_number, fmt_euro
from config import GOLD, GOLD2, TEAL, CORAL, PURPLE, DARK_BG, GRID_COL, TEXT_COL


def render(filters_key: str):
    with st.spinner("Chargement…"):
        kpi   = load_kpi_global(filters_key)
        df_ct = load_contrats_salaire_stats(filters_key)
        df_an = load_anciennete_salaire_stats(filters_key)

    row = kpi.iloc[0]

    # ── KPI row 1 ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Total offres actives et clôturées", fmt_number(row.total_offres), "toutes sources", GOLD), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Salaire moyen", fmt_euro(row.salaire_moyen), "brut annuel estimé", TEAL), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Entreprises", fmt_number(row.nb_entreprises), "recruteurs distincts", CORAL), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Offres actives", fmt_number(row.offres_actives), "statut = active", PURPLE), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── KPI row 2 ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Offres < 7 jours",  fmt_number(row.offres_7j),  "publiées cette semaine", GOLD2), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Offres < 30 jours", fmt_number(row.offres_30j), "publiées ce mois",       GOLD2), unsafe_allow_html=True)
    with c3:
        # Durée moyenne d'ouverture à la place des offres < 90j
        duree = row.get("duree_moyenne_jours") if hasattr(row, "get") else getattr(row, "duree_moyenne_jours", None)
        import math
        duree_val = "—" if duree is None or (isinstance(duree, float) and math.isnan(duree)) else f"{float(duree):.0f} j"
        st.markdown(kpi_card("Durée moy. ouverture", duree_val, "jours entre publication et clôture", GOLD2), unsafe_allow_html=True)
    with c4:
        st.markdown(
            kpi_card(
                "Offres clôturées / supprimées",
                fmt_number(row.offres_pourvues),
                "fermées sur les 14 derniers jours",
                GOLD2,
            ),
            unsafe_allow_html=True,
        )


    # ══════════════════════════════════════════════════════════════════════
    # FLUX DE PUBLICATION
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Flux de publication</div>', unsafe_allow_html=True)

    col_toggle, _ = st.columns([2, 5])
    with col_toggle:
        granularite = st.radio(
            "Granularité",
            options=["Par jour", "Par semaine"],
            horizontal=True,
            label_visibility="collapsed",
            key="flux_granularite",
        )

    par_semaine = "semaine" in granularite

    if par_semaine:
        df_flux = load_offres_par_semaine(filters_key)
    else:
        df_flux = load_offres_par_jour(filters_key)

    if not df_flux.empty:
        if par_semaine:
            x_vals = df_flux["label_semaine"].tolist()
            y_vals = df_flux["nb"].tolist()
            titre  = f"Offres publiées par semaine ({len(x_vals)} semaines)"
        else:
            x_vals = [str(j) for j in df_flux["jour"].tolist()]
            y_vals = df_flux["nb"].tolist()
            titre  = f"Offres publiées par jour ({len(x_vals)} jours)"

        nb_points  = len(x_vals)
        # Nb de barres visibles à la fois dans la fenêtre (max 30)
        nb_visible = min(nb_points, 30)

        fig_flux = go.Figure(go.Bar(
            x=x_vals,
            y=y_vals,
            marker=dict(
                color=y_vals,
                colorscale=[[0, "#2a2410"], [0.5, "#8a6020"], [1, GOLD]],
                showscale=False,
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>%{y:,} offres<extra></extra>",
        ))

        layout = base_layout(titre)
        layout.update(dict(
            height=320,
            bargap=0.25,
            xaxis=dict(
                **layout.get("xaxis", {}),
                tickangle=-45,
                tickfont=dict(size=10),
                type="category",
                # Range initial : afficher les nb_visible dernières barres
                range=[nb_points - nb_visible - 0.5, nb_points - 0.5],
                rangeslider=dict(
                    visible=True,
                    thickness=0.06,
                    bgcolor="#1a1c24",
                    bordercolor=GRID_COL,
                    borderwidth=1,
                ),
            ),
        ))
        fig_flux.update_layout(**layout)
        st.plotly_chart(fig_flux, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # RÉPARTITION
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Répartition des offres</div>', unsafe_allow_html=True)

    def _add_candles(fig: go.Figure, df: pd.DataFrame, y_col: str, xaxis: str = "x2") -> go.Figure:
        """Ajoute des bougies salaires (P25-P75 + médiane + min-max)."""
        required = ("salaire_min", "salaire_p25", "salaire_moyen", "salaire_p75", "salaire_max")
        if not df.empty and all(c in df.columns for c in required):
            d = df.copy()
        else:
            return fig

        # Barres d'erreur min→max
        line_mask = d["salaire_min"].notna() & d["salaire_max"].notna()
        d_line = d[line_mask]
        for _, row in d_line.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["salaire_min"], row["salaire_max"]],
                y=[row[y_col], row[y_col]],
                mode="lines",
                line=dict(color="rgba(212,168,75,0.40)", width=0.8),
                xaxis=xaxis,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Corps de la bougie (P25 → P75)
        bar_mask = d["salaire_p25"].notna() & d["salaire_p75"].notna()
        d_bar = d[bar_mask]
        if not d_bar.empty:
            fig.add_trace(go.Bar(
                x=(d_bar["salaire_p75"] - d_bar["salaire_p25"]).clip(lower=0),
                y=d_bar[y_col],
                orientation="h",
                base=d_bar["salaire_p25"],
                marker=dict(color="rgba(212,168,75,0.45)", line=dict(color="#d4a84b", width=0.8)),
                xaxis=xaxis,
                name="Sal. p25–p75",
                customdata=d_bar["salaire_p75"],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "p25 : %{base:,.0f} €<br>"
                    "p75 : %{customdata:,.0f} €<extra></extra>"
                ),
            ))

        # Marqueur médiane / moyenne
        moy_mask = d["salaire_moyen"].notna()
        d_moy = d[moy_mask]
        if not d_moy.empty:
            fig.add_trace(go.Scatter(
                x=d_moy["salaire_moyen"],
                y=d_moy[y_col],
                mode="markers",
                marker=dict(color="#d4a84b", size=7, symbol="diamond"),
                xaxis=xaxis,
                name="Sal. moyen",
                hovertemplate="<b>%{y}</b><br>Sal. moyen : %{x:,.0f} €<extra></extra>",
            ))

        return fig
    hide_non_precise = st.checkbox(
        "Masquer les valeurs 'Non précisé'",
        value=True,
        key="hide_non_precise_repartition",
    )

    def _is_non_precise(v) -> bool:
        if v is None:
            return True
        s = str(v).strip().lower()
        return s in {"", "unknown", "non precise", "non précisé", "non renseigne", "non renseigné", "n/a", "na"}

    if hide_non_precise:
        if not df_ct.empty:
            df_ct = df_ct[~df_ct["contract_type"].apply(_is_non_precise)]
        if not df_an.empty:
            df_an = df_an[~df_an["experience_level"].apply(_is_non_precise)]

    col_l, col_r = st.columns(2)

    with col_l:
        if not df_ct.empty:
            fig_ct = go.Figure(go.Bar(
                x=df_ct["nb"], y=df_ct["contract_type"], orientation="h",
                marker=dict(color=df_ct["nb"], 
                            colorscale=[[0, "#2a2410"], [1, PURPLE]], 
                            showscale=False),
                hovertemplate="<b>%{y}</b><br>%{x:,} offres<extra></extra>",
            ))
            fig_ct = _add_candles(fig_ct, df_ct, y_col="contract_type", xaxis="x2")

            layout_ct = base_layout("Par type de contrat")
            layout_ct.update(
                height=320,
                barmode="overlay",
                xaxis2=dict(
                    title="Salaire (€)",
                    overlaying="x",
                    side="top",
                    showgrid=False,
                ),
            )
            fig_ct.update_layout(**layout_ct)
            fig_ct.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_ct, use_container_width=True)

    with col_r:
        if not df_an.empty:
            fig_an = go.Figure(go.Bar(
                x=df_an["nb"], y=df_an["experience_level"], orientation="h",
                marker=dict(color=df_an["nb"], colorscale=[[0, "#2a1a10"], [1, PURPLE]], showscale=False),
                hovertemplate="<b>%{y}</b><br>%{x:,} offres<extra></extra>",
            ))
            fig_an = _add_candles(fig_an, df_an, y_col="experience_level", xaxis="x2")

            layout_an = base_layout("Par niveau d'expérience")
            layout_an.update(
                height=320,
                barmode="overlay",
                xaxis2=dict(
                    title="Salaire (€)",
                    overlaying="x",
                    side="top",
                    showgrid=False,
                ),
            )
            fig_an.update_layout(**layout_an)
            fig_an.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_an, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # CODES NAF
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Top secteurs par code NAF - Nombre offres</div>', unsafe_allow_html=True)

    naf_col1, naf_col2, naf_col3 = st.columns([3, 2, 2])
    with naf_col1:
        regions_list = ["Toutes"] + load_regions_list(filters_key)
        sel_region_naf = st.selectbox("Filtrer par région", regions_list, index=0, key="naf_region")
    with naf_col2:
        top_n = st.select_slider("Nombre de secteurs", options=[10, 15, 20, 30, 50], value=20, key="naf_top_n")
    with naf_col3:
        tri_mode = st.session_state.get("tri_offres_vs_salaire", "Nombre d'offres")

    with st.spinner("Chargement NAF…"):
        df_naf = load_naf_par_region_salaire_stats(
            region=sel_region_naf if sel_region_naf != "Toutes" else "",
            top_n=top_n,
            filters_key=filters_key + sel_region_naf + str(top_n),
        )

    if not df_naf.empty:
        # Ré-ordonne selon la métrique choisie dans la sidebar,
        # mais le graphique NAF reste "Nombre d'offres" + bougies salaire.
        if tri_mode == "Salaire moyen":
            df_naf = df_naf.sort_values(["salaire_moyen", "nb_offres"], ascending=[False, False], na_position="last")
        else:
            df_naf = df_naf.sort_values(["nb_offres", "salaire_moyen"], ascending=[False, False], na_position="last")
        df_naf["label"] = df_naf.apply(
            lambda r: f"{r['naf_code']} — {str(r['naf_label'])[:42]}{'…' if len(str(r['naf_label'])) > 42 else ''}",
            axis=1,
        )
        x_vals    = df_naf["nb_offres"]
        color_max = GOLD
        color_min = "#2a2410"
        x_title   = "Nombre d'offres"
        titre_naf = f"Top {top_n} codes NAF — {sel_region_naf}"

        fig_naf = go.Figure(go.Bar(
            x=x_vals, y=df_naf["label"], orientation="h",
            marker=dict(color=x_vals, colorscale=[[0, color_min], [1, color_max]], showscale=False),
            customdata=df_naf[["nb_offres", "salaire_moyen", "naf_code"]].values,
            hovertemplate="<b>%{y}</b><br>Offres : %{customdata[0]:,}<br>Sal. moy. : %{customdata[1]:,.0f} €<extra></extra>",
        ))
        layout_naf = base_layout(titre_naf)
        layout_naf.update(dict(
            height=max(320, len(df_naf) * 28 + 60),
            xaxis=dict(**layout_naf.get("xaxis", {}), title_text=x_title, tickformat=","),
        ))
        layout_naf.update(dict(
            barmode="overlay",
            xaxis2=dict(
                title="Salaire annuel (€)",
                overlaying="x",
                side="top",
                showgrid=False,
            ),
        ))
        fig_naf.update_layout(**layout_naf)
        fig_naf = _add_candles(fig_naf, df_naf, y_col="label", xaxis="x2")
        fig_naf.update_yaxes(autorange="reversed", tickfont=dict(size=11))
        st.plotly_chart(fig_naf, use_container_width=True)
    else:
        st.info("Aucune donnée NAF disponible pour cette sélection.")
