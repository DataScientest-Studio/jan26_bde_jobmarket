"""
Sidebar de filtres — région / département / ville / secteur / poste / contrat / dates.
Les filtres sélectionnés sont stockés dans st.session_state["active_filters"].
"""
import datetime
import streamlit as st
from utils.queries import load_filter_options, search_villes, search_rome, search_entreprises


def render_sidebar() -> str:
    """
    Affiche la sidebar et met à jour st.session_state["active_filters"].
    Retourne une clé de cache (str) représentant l'état courant des filtres,
    à passer aux fonctions @st.cache_data pour invalider le cache si besoin.
    """
    with st.sidebar:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                background: #12141a;
                border-right: 1px solid #24262e;
            }
            [data-testid="stSidebar"] .sidebar-title {
                font-family: 'DM Serif Display', serif;
                font-size: 1.1rem;
                color: #d4c9b0;
                margin-bottom: 0.2rem;
            }
            [data-testid="stSidebar"] .sidebar-section {
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: #5a5a64;
                margin: 1.2rem 0 0.4rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-title">Filtres</div>', unsafe_allow_html=True)

        # Chargement des options disponibles
        with st.spinner("Chargement des filtres…"):
            opts = load_filter_options()

        # ── Géographie ────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Géographie</div>', unsafe_allow_html=True)

        sel_regions = st.multiselect(
            "Région(s)",
            options=opts["regions"],
            default=[],
            placeholder="Toutes les régions",
        )

        # Filtrer les départements selon la/les région(s) choisie(s)
        dep_df = opts["departements"]
        if sel_regions:
            dep_df = dep_df[dep_df["nom_region"].isin(sel_regions)]
        dep_list = dep_df["nom_departement"].tolist()

        sel_departements = st.multiselect(
            "Département(s)",
            options=dep_list,
            default=[],
            placeholder="Tous les départements",
        )

        # ── Tri (offres vs salaire moyen) ────────────────────────────────
        st.markdown('<div class="sidebar-section">Tri</div>', unsafe_allow_html=True)
        tri_offres_vs_salaire = st.radio(
            "Trier par",
            options=["Nombre d'offres", "Salaire moyen"],
            index=0,
            horizontal=True,
            key="tri_offres_vs_salaire",
        )

        # ── Recherche ville dynamique ──────────────────────────────────
        ville_search = st.text_input(
            "Rechercher une ville",
            value=st.session_state.get("ville_search_input", ""),
            placeholder="ex: Par, Lyon, Mar…",
            key="ville_search_input",
        )

        # Résultats de la recherche (dès 2 caractères)
        ville_results = []
        if len(ville_search.strip()) >= 2:
            ville_results = search_villes(
                prefix=ville_search.strip(),
                regions=tuple(sel_regions),
                departements=tuple(sel_departements),
            )
            if not ville_results:
                st.caption("Aucune ville trouvée.")

        # Villes déjà sélectionnées (persistées en session)
        if "sel_villes" not in st.session_state:
            st.session_state["sel_villes"] = []

        # Afficher les résultats comme options à cocher
        if ville_results:
            st.caption(f"{len(ville_results)} ville(s) trouvée(s) :")
            for ville in ville_results:
                already = ville in st.session_state["sel_villes"]
                if st.checkbox(ville, value=already, key=f"ville_cb_{ville}"):
                    if ville not in st.session_state["sel_villes"]:
                        st.session_state["sel_villes"].append(ville)
                else:
                    if ville in st.session_state["sel_villes"]:
                        st.session_state["sel_villes"].remove(ville)

        # Afficher les villes sélectionnées avec possibilité de retrait
        sel_villes = st.session_state["sel_villes"]
        if sel_villes:
            st.caption("Villes sélectionnées :")
            to_remove = []
            for v in list(sel_villes):
                col_v, col_x = st.columns([5, 1])
                col_v.markdown(
                    f'<span style="font-size:0.82rem;color:#d4a84b">📍 {v}</span>',
                    unsafe_allow_html=True,
                )
                if col_x.button("✕", key=f"rm_{v}", help=f"Retirer {v}"):
                    to_remove.append(v)
            for v in to_remove:
                st.session_state["sel_villes"].remove(v)
                st.rerun()

        # ── Secteur & poste ───────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Métier</div>', unsafe_allow_html=True)

        # ── Recherche ROME dynamique ───────────────────────────────────────
        rome_search = st.text_input(
            "Rechercher un secteur ROME",
            placeholder="ex: informatique, A1101, comptab…",
            key="rome_search_input",
        )

        rome_results = []
        if len(rome_search.strip()) >= 2:
            rome_results = search_rome(rome_search.strip())
            if not rome_results:
                st.caption("Aucun code ROME trouvé.")

        if "sel_secteurs" not in st.session_state:
            st.session_state["sel_secteurs"] = []  # liste de dicts {rome_code, rome_label}

        sel_codes = [r["rome_code"] for r in st.session_state["sel_secteurs"]]

        if rome_results:
            st.caption(f"{len(rome_results)} résultat(s) :")
            for item in rome_results:
                already = item["rome_code"] in sel_codes
                label = f"{item['rome_code']} — {item['rome_label']}"
                if st.checkbox(label, value=already, key=f"rome_cb_{item['rome_code']}"):
                    if item["rome_code"] not in sel_codes:
                        st.session_state["sel_secteurs"].append(item)
                        sel_codes.append(item["rome_code"])
                else:
                    if item["rome_code"] in sel_codes:
                        st.session_state["sel_secteurs"] = [
                            r for r in st.session_state["sel_secteurs"]
                            if r["rome_code"] != item["rome_code"]
                        ]
                        sel_codes = [r["rome_code"] for r in st.session_state["sel_secteurs"]]

        # Affichage des secteurs sélectionnés avec retrait individuel
        sel_secteurs_list = st.session_state["sel_secteurs"]
        if sel_secteurs_list:
            st.caption("✓ Secteurs sélectionnés :")
            to_remove = []
            for item in list(sel_secteurs_list):
                col_v, col_x = st.columns([5, 1])
                col_v.markdown(
                    f'<span style="font-size:0.78rem;color:#d4a84b">💼 {item["rome_code"]} {item["rome_label"]}</span>',
                    unsafe_allow_html=True,
                )
                if col_x.button("✕", key=f"rm_rome_{item['rome_code']}", help=f"Retirer {item['rome_label']}"):
                    to_remove.append(item["rome_code"])
            for code in to_remove:
                st.session_state["sel_secteurs"] = [
                    r for r in st.session_state["sel_secteurs"] if r["rome_code"] != code
                ]
                st.rerun()

        # Les secteurs actifs = liste des rome_label pour _build_where
        sel_secteurs = [r["rome_label"] for r in st.session_state["sel_secteurs"]]

        sel_poste = st.text_input(
            "Recherche poste",
            value=st.session_state.get("poste_search_input", ""),
            placeholder="ex: développeur, comptable…",
            key="poste_search_input",
        )

        # ── Entreprises ───────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Entreprises</div>', unsafe_allow_html=True)
        entreprise_search = st.text_input(
            "Rechercher une entreprise",
            value=st.session_state.get("entreprise_search_input", ""),
            placeholder="ex: Air, Total, Orange…",
            key="entreprise_search_input",
        )

        entreprise_results = []
        if len(entreprise_search.strip()) >= 2:
            entreprise_results = search_entreprises(entreprise_search.strip())
            if not entreprise_results:
                st.caption("Aucune entreprise trouvée.")

        if "sel_entreprises" not in st.session_state:
            st.session_state["sel_entreprises"] = []

        if entreprise_results:
            st.caption(f"{len(entreprise_results)} entreprise(s) trouvée(s) :")
            for ent in entreprise_results:
                already = ent in st.session_state["sel_entreprises"]
                if st.checkbox(ent, value=already, key=f"entreprise_cb_{ent}"):
                    if ent not in st.session_state["sel_entreprises"]:
                        st.session_state["sel_entreprises"].append(ent)
                else:
                    if ent in st.session_state["sel_entreprises"]:
                        st.session_state["sel_entreprises"].remove(ent)

        sel_entreprises = st.session_state["sel_entreprises"]
        if sel_entreprises:
            st.caption("Entreprises sélectionnées :")
            to_remove_ent = []
            for ent in list(sel_entreprises):
                col_v, col_x = st.columns([5, 1])
                col_v.markdown(
                    f'<span style="font-size:0.78rem;color:#d4a84b">🏢 {ent}</span>',
                    unsafe_allow_html=True,
                )
                if col_x.button("✕", key=f"rm_ent_{ent}", help=f"Retirer {ent}"):
                    to_remove_ent.append(ent)
            for ent in to_remove_ent:
                st.session_state["sel_entreprises"].remove(ent)
                st.rerun()

        # ── Contrat ───────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Contrat</div>', unsafe_allow_html=True)

        sel_contrats = st.multiselect(
            "Type(s) de contrat",
            options=opts["contrats"],
            default=[],
            placeholder="Tous les contrats",
        )

        # ── Dates de publication ──────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Publication</div>', unsafe_allow_html=True)

        date_rapide = st.radio(
            "Période rapide",
            options=["Tout", "7 jours", "30 jours", "90 jours", "Personnalisé"],
            index=0,
            horizontal=False,
        )

        today = datetime.date.today()
        date_debut = None
        date_fin = None

        if date_rapide == "7 jours":
            date_debut = today - datetime.timedelta(days=7)
        elif date_rapide == "30 jours":
            date_debut = today - datetime.timedelta(days=30)
        elif date_rapide == "90 jours":
            date_debut = today - datetime.timedelta(days=90)
        elif date_rapide == "Personnalisé":
            col1, col2 = st.columns(2)
            with col1:
                date_debut = st.date_input("Du", value=today - datetime.timedelta(days=30))
            with col2:
                date_fin = st.date_input("Au", value=today)

        # ── Bouton reset ──────────────────────────────────────────────────
        st.markdown("---")
        if st.button("↺ Réinitialiser", use_container_width=True):
            st.session_state["active_filters"] = {}
            st.session_state["sel_villes"] = []
            st.session_state["sel_secteurs"] = []
            st.session_state["sel_entreprises"] = []
            st.rerun()

        # ── Résumé des filtres actifs ─────────────────────────────────────
        active_filters = {
            "regions":      sel_regions,
            "departements": sel_departements,
            "villes":       sel_villes,
            "secteurs":     sel_secteurs,
            "entreprises":  sel_entreprises,
            "postes":       sel_poste.strip(),
            "contrats":     sel_contrats,
            "date_debut":   str(date_debut) if date_debut else None,
            "date_fin":     str(date_fin) if date_fin else None,
            "tri_offres_vs_salaire": tri_offres_vs_salaire,
        }
        st.session_state["active_filters"] = active_filters

        # Compteur de filtres actifs
        nb_actifs = sum([
            len(sel_regions),
            len(sel_departements),
            len(sel_villes),
            len(sel_secteurs),
            len(sel_entreprises),
            1 if sel_poste.strip() else 0,
            len(sel_contrats),
            1 if date_debut else 0,
        ])
        if nb_actifs:
            st.markdown(
                f'<div style="text-align:center;color:#d4a84b;font-size:0.8rem;">'
                f'✓ {nb_actifs} filtre{"s" if nb_actifs > 1 else ""} actif{"s" if nb_actifs > 1 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Clé de cache = repr des filtres (pour invalider @st.cache_data)
    return str(sorted(active_filters.items()))
