"""
Tab Carte — carte Folium interactive avec bulles proportionnelles.
Clic sur une bulle → st_folium retourne last_object_clicked → panneau détail.
"""
import json, math, os
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from utils.queries import load_carte_regions, load_carte_departements, load_top_contrats_zone
from utils.helpers import fmt_number, fmt_euro
from config import GOLD, TEAL, CORAL, PURPLE, TEXT_COL

_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOJSON_REGIONS = os.path.join(_DIR, "regions.geojson")
GEOJSON_DEPS    = os.path.join(_DIR, "departements.geojson")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _centroid(coordinates):
    pts = []
    def _flat(c):
        if isinstance(c[0], (int, float)): pts.append(c)
        else:
            for s in c: _flat(s)
    _flat(coordinates)
    if not pts: return 46.5, 2.0
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return sum(lats)/len(lats), sum(lons)/len(lons)   # lat, lon pour folium

def _safe(v):
    if v is None: return None
    try:
        f = float(v); return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

def _radius_px(nb, vmin, vmax, r_min, r_max):
    """Rayon en pixels pour folium CircleMarker."""
    if vmax == vmin:
        return (r_min + r_max) // 2
    # nb peut être < vmin (zones sans données => nb=0). On clamp pour éviter sqrt(<0).
    denom = (vmax - vmin)
    ratio = 0.0 if denom == 0 else (nb - vmin) / denom
    ratio = 0.0 if ratio < 0 else (1.0 if ratio > 1 else ratio)
    t = ratio ** 0.5
    return int(r_min + t * (r_max - r_min))


# ── Construction de la carte Folium ───────────────────────────────────────────

def _build_map(geojson_path, df, is_region):
    with open(geojson_path) as f:
        geo = json.load(f)

    # Pour les départements en particulier, les libellés peuvent diverger (accents, tirets…).
    # On privilégie donc l'appariement par code, bien plus robuste.
    if df is None or df.empty:
        df_idx_code = None
        df_idx_name = None
        vmin = 0.0
        vmax = 0.0
    else:
        df_idx_code = df.set_index(df["code"].astype(str).str.strip().str.upper())
        df_idx_name = df.set_index(df["nom"].astype(str).str.strip().str.lower())
        vmin   = float(df["nb_offres"].min())
        vmax   = float(df["nb_offres"].max())
    r_min  = 8  if is_region else 5
    r_max  = 35 if is_region else 22

    m = folium.Map(
        location=[46.6, 2.5],
        zoom_start=5,
        tiles=None,
        prefer_canvas=True,
    )

    # Fond sombre : tuiles CartoDB Dark Matter
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap © CARTO",
        name="Dark Matter",
        max_zoom=19,
    ).add_to(m)

    # Fond polygones (bordures uniquement)
    folium.GeoJson(
        geo,
        style_function=lambda _: {
            "fillColor":   "transparent",
            "color":       "#3a3d4a",
            "weight":      1.2,
            "fillOpacity": 0,
        },
        highlight_function=lambda _: {
            "color":  "#d4a84b",
            "weight": 2,
        },
    ).add_to(m)

    # Bulles par zone
    for feat in geo["features"]:
        nom_geo = feat["properties"].get("nom", "")
        nom_key = nom_geo.strip().lower()
        code    = feat["properties"].get("code", "")
        code_key = str(code).strip().upper()
        lat, lon = _centroid(feat["geometry"]["coordinates"])

        row = None
        if df_idx_code is not None and code_key in df_idx_code.index:
            row = df_idx_code.loc[code_key]
        elif df_idx_name is not None and nom_key in df_idx_name.index:
            row = df_idx_name.loc[nom_key]

        if row is not None:
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            nb      = int(row["nb_offres"])
            nb_ent  = int(row.get("nb_entreprises", 0))
            sal_moy = _safe(row.get("salaire_moyen"))
            sal_med = _safe(row.get("salaire_mediane"))
            off30   = int(row.get("offres_30j", 0))
            nom_reg = row.get("nom_region", "") if not is_region else ""
        else:
            nb = 0; nb_ent = 0; sal_moy = None; sal_med = None; off30 = 0; nom_reg = ""

        radius  = _radius_px(nb, vmin, vmax, r_min, r_max)
        if vmax > vmin:
            ratio = (nb - vmin) / (vmax - vmin)
            ratio = 0.0 if ratio < 0 else (1.0 if ratio > 1 else ratio)
            t = ratio ** 0.5
        else:
            t = 0.5
        opacity = 0.25 + t * 0.55    # 0.25 → 0.80
        sal_str = fmt_euro(sal_moy)
        nb_fmt  = fmt_number(nb)

        # Tooltip HTML au survol
        tooltip_html = f"""
        <div style="font-family:DM Sans,sans-serif;font-size:13px;
                    color:#e8e6e0;min-width:180px">
            <b style="color:#d4a84b;font-size:14px">{nom_geo}</b>
            <span style="color:#5a5a64;font-size:11px"> {code}</span><br>
            <table style="margin-top:5px;border-collapse:collapse;width:100%">
                <tr><td style="color:#6b6a66;padding:1px 6px 1px 0">Offres</td>
                    <td style="font-weight:600">{nb_fmt}</td></tr>
                <tr><td style="color:#6b6a66;padding:1px 6px 1px 0">Entreprises</td>
                    <td style="font-weight:600">{nb_ent}</td></tr>
                <tr><td style="color:#6b6a66;padding:1px 6px 1px 0">Sal. moyen</td>
                    <td style="font-weight:600;color:#30b439">{sal_str}</td></tr>
                <tr><td style="color:#6b6a66;padding:1px 6px 1px 0">30 derniers j.</td>
                    <td style="font-weight:600;color:#7a6aad">{off30}</td></tr>
            </table>
            <div style="color:#5a5a64;font-size:10px;margin-top:4px">
                Cliquer pour le détail
            </div>
        </div>
        """

        # Popup (transmis à st_folium via last_object_clicked_popup)
        popup_content = nom_geo   # on utilise juste le nom pour identifier la zone

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#d4a84b",
            weight=2,
            fill=True,
            fill_color="#d4a84b",
            fill_opacity=opacity,
            tooltip=folium.Tooltip(
                tooltip_html,
                sticky=True,
                style=(
                    "background:#1a1c24;"
                    "border:1px solid #d4a84b55;"
                    "border-radius:10px;"
                    "padding:10px 14px;"
                    "box-shadow:0 4px 20px rgba(0,0,0,0.6);"
                ),
            ),
            popup=folium.Popup(popup_content, max_width=1),
        ).add_to(m)

        # Label chiffre au centre
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-family:DM Sans,sans-serif;'
                     f'font-size:{max(9, radius//2)}px;font-weight:700;'
                     f'color:#f0ede6;text-shadow:0 1px 3px rgba(0,0,0,0.8);'
                     f'white-space:nowrap;pointer-events:none">{nb_fmt}</div>',
                icon_size=(60, 20),
                icon_anchor=(30, 10),
            ),
        ).add_to(m)

    return m


# ── Rendu principal ───────────────────────────────────────────────────────────

def render(_filters_key: str = ""):
    with st.spinner("Chargement des données carte…"):
        df_reg = load_carte_regions()
        df_dep = load_carte_departements()

    # ── Barre de contrôles ────────────────────────────────────────────────
    ctrl_col, info_col = st.columns([2, 5])
    with ctrl_col:
        niveau = st.radio(
            "Niveau", options=["Régions", "Départements"],
            horizontal=True, label_visibility="collapsed",
        )

    is_region = "Région" in niveau
    df_active = df_reg if is_region else df_dep
    geojson_path = GEOJSON_REGIONS if is_region else GEOJSON_DEPS
    sel_key   = "carte_sel_r" if is_region else "carte_sel_d"

    with info_col:
        total = int(df_active["nb_offres"].sum())
        top   = df_active.iloc[0]
        st.markdown(
            f'<div style="padding-top:0.45rem;font-size:0.85rem;color:{TEXT_COL}">'
            f'<span style="color:{GOLD};font-weight:700">{fmt_number(total)}</span>'
            f' offres &nbsp;·&nbsp; Zone n°1 : '
            f'<span style="color:{TEAL};font-weight:600">{top["nom"]}</span>'
            f' ({fmt_number(int(top["nb_offres"]))} offres)'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Carte + panneau détail ────────────────────────────────────────────
    col_map, col_detail = st.columns([4, 1])

    with col_map:
        m = _build_map(geojson_path, df_active, is_region)

        result = st_folium(
            m,
            use_container_width=True,
            height=680,
            returned_objects=["last_object_clicked_popup"],
        )

        # ── Lecture du clic ───────────────────────────────────────────────
        clicked_nom = None
        try:
            popup_val = result.get("last_object_clicked_popup")
            if popup_val and isinstance(popup_val, str) and popup_val.strip():
                clicked_nom = popup_val.strip()
        except Exception:
            pass

        if clicked_nom and clicked_nom in df_active["nom"].values:
            st.session_state[sel_key] = clicked_nom

        _render_legend(df_active, is_region)

    with col_detail:
        _render_detail_panel(df_active, is_region, sel_key)


# ── Panneau détail ────────────────────────────────────────────────────────────

def _render_detail_panel(df, is_region, sel_key):
    zones = df["nom"].sort_values().tolist()
    prev  = st.session_state.get(sel_key, zones[0] if zones else "")
    idx   = zones.index(prev) if prev in zones else 0

    zone = st.selectbox(
        "Zone", zones, index=idx, key=sel_key,
        label_visibility="collapsed",
    )
    if not zone: return

    row = df[df["nom"] == zone]
    if row.empty: return
    row = row.iloc[0]

    nb      = int(row["nb_offres"])
    nb_ent  = int(row.get("nb_entreprises", 0))
    sal_moy = _safe(row.get("salaire_moyen"))
    sal_med = _safe(row.get("salaire_mediane"))
    off30   = int(row.get("offres_30j", 0))
    total   = int(df["nb_offres"].sum())
    pct     = nb / total * 100 if total else 0
    rang    = int(df["nb_offres"].rank(ascending=False).loc[df["nom"] == zone].iloc[0])

    def card(label, value, color=GOLD, sub=""):
        sub_html = (
            f'<div style="font-size:0.7rem;color:#4a4a52;margin-top:0.15rem">{sub}</div>'
            if sub else ""
        )
        return (
            f'<div style="background:#16181f;border:1px solid #24262e;border-radius:8px;'
            f'padding:0.7rem 0.8rem;margin-bottom:0.5rem">'
            f'<div style="font-size:0.62rem;color:#5a5a64;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:0.15rem">{label}</div>'
            f'<div style="font-size:1.35rem;font-family:DM Serif Display,serif;'
            f'color:{color};line-height:1.1">{value}</div>'
            f'{sub_html}</div>'
        )

    st.markdown(
        f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
        f'text-transform:uppercase;color:{GOLD};margin-bottom:0.5rem;margin-top:0.2rem">'
        f'📊 {zone}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        card("Offres", fmt_number(nb), GOLD, f"{pct:.1f}% national · rang #{rang}") +
        card("Sal. moyen", fmt_euro(sal_moy), TEAL, f"Médiane : {fmt_euro(sal_med)}") +
        card("Entreprises", fmt_number(nb_ent), CORAL) +
        card("30 derniers j.", fmt_number(off30), PURPLE),
        unsafe_allow_html=True,
    )

    if not is_region:
        nom_reg = row.get("nom_region", "")
        if nom_reg and isinstance(nom_reg, str):
            st.markdown(
                f'<div style="font-size:0.72rem;color:#4a4a52;margin-bottom:0.6rem">'
                f'Région : <span style="color:#9a9890">{nom_reg}</span></div>',
                unsafe_allow_html=True,
            )

    try:
        df_ct = load_top_contrats_zone(
            "region" if is_region else "departement", zone
        )
        if not df_ct.empty:
            st.markdown(
                f'<div style="font-size:0.62rem;color:#5a5a64;text-transform:uppercase;'
                f'letter-spacing:0.08em;margin-bottom:0.4rem">Top contrats</div>',
                unsafe_allow_html=True,
            )
            ct_max = int(df_ct["nb"].max())
            bars = ""
            for _, ct in df_ct.iterrows():
                w = int(ct["nb"] / ct_max * 100)
                bars += (
                    f'<div style="margin-bottom:0.35rem">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.7rem;color:#9a9890;margin-bottom:2px">'
                    f'<span style="white-space:nowrap;overflow:hidden;'
                    f'text-overflow:ellipsis;max-width:75%">{ct["contract_type"]}</span>'
                    f'<span style="color:{GOLD}">{fmt_number(int(ct["nb"]))}</span></div>'
                    f'<div style="background:#24262e;border-radius:3px;height:3px">'
                    f'<div style="background:{GOLD};width:{w}%;'
                    f'height:3px;border-radius:3px"></div></div></div>'
                )
            st.markdown(bars, unsafe_allow_html=True)
    except Exception:
        pass


# ── Légende ───────────────────────────────────────────────────────────────────

def _render_legend(df, is_region):
    vmin = int(df["nb_offres"].min())
    vmax = int(df["nb_offres"].max())
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.8rem;'
        f'font-size:0.73rem;color:#6b6a66;margin-top:0.3rem">'
        f'Taille des cercles ∝ volume d\'offres &nbsp;'
        f'<span style="color:{GOLD}">⬤</span> {fmt_number(vmax)} (max)'
        f'&nbsp;<span style="color:#d4a84b44">⬤</span> {fmt_number(vmin)} (min)'
        f'&nbsp;<span style="color:#4a4a52">{len(df)} zones</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
