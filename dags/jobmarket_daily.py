"""
DAG quotidien — Pipeline jobmarket complet.

Chaîne d'exécution :
    ingest_ft → ingest_wttj → [normalize_wttj, normalize_ft] → merge
    → status_tracking → status_evolution → load_star_schema

Notes :
- FT est ingéré en premier (plus rapide que WTTJ en synchrone).
- normalize_wttj et normalize_ft s'exécutent en parallèle une fois les
  deux ingest terminés.
- Toutes les tâches appellent l'API en mode synchrone (background=False) :
  Airflow attend la fin réelle du traitement avant de passer à la suivante.

Mode debug :
    Déclencher via "Trigger DAG w/ config" → {"debug": true}
    → FT limité à 2 codes ROME, WTTJ limité à 50 offres.
    Les étapes suivantes s'exécutent normalement sur ces données réduites.
"""

from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constantes — modifier ici pour ajuster le comportement sans toucher la logique
# ---------------------------------------------------------------------------

# Limites du mode debug
DEBUG_FT_MAX_ROME_CODES  = 2   # nombre de codes ROME ingérés en mode debug
DEBUG_WTTJ_MAX_JOBS      = 50  # nombre d'offres WTTJ ingérées en mode debug
DEBUG_WTTJ_PART_SIZE     = 50  # taille du chunk JSONL = max_jobs → 1 seul fichier généré

# Endpoints de l'API jobmarket
API_INGEST_FT           = "/ingest/france-travail-offers"
API_INGEST_WTTJ         = "/ingest/welcome-to-jungle"
API_NORMALIZE_WTTJ      = "/data/normalize-wttj-jobs"     # default=False
API_NORMALIZE_FT        = "/data/normalize-ft-jobs"       # default=False
API_MERGE               = "/data/merge-datasets"          # default=False
API_STATUS_TRACKING     = "/data/status-tracking"         # default=False
API_STATUS_EVOLUTION    = "/data/status-evolution"                 # default=False
API_LOAD_STAR_SCHEMA    = "/gold/load-star-schema?source_mode=auto"  # default=False

# ---------------------------------------------------------------------------
# Pas besoin de charger les dépendances dans le projet local, Airflow est géré en Docker

# DAG : conteneur du pipeline. Définit le scheduling, les métadonnées
# et les paramètres passables au déclenchement.
from airflow import DAG

# Param : déclare un paramètre typé visible dans l'UI "Trigger DAG w/ config".
from airflow.models.param import Param

# PythonOperator : exécute une fonction Python comme tâche Airflow.
# Utilisé ici pour construire dynamiquement l'URL d'endpoint selon le mode debug.
from airflow.operators.python import PythonOperator

# HttpOperator : envoie une requête HTTP et vérifie la réponse.
# Chaque appel à l'API jobmarket est une tâche HttpOperator.
from airflow.providers.http.operators.http import HttpOperator

# Label : affiche un libellé sur une flèche dans la vue graphique du DAG.
from airflow.utils.edgemodifier import Label

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _notify_slack(text: str) -> None:
    """Envoie un message Slack via Incoming Webhook.

    Ne fait rien si SLACK_WEBHOOK_URL n'est pas déclarée dans l'environnement —
    permet de déployer sans Slack configuré sans erreur.
    """
    import os, requests
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    requests.post(url, json={"text": text}, timeout=10)


def _on_failure(context) -> None:
    """Appelé par Airflow sur échec ou execution_timeout d'une tâche."""
    ti = context["task_instance"]
    exc = context.get("exception", "")
    _notify_slack(
        f":x: *Échec* — DAG `{ti.dag_id}` — tâche `{ti.task_id}`\n```{exc}```"
    )


def _on_task_success(context) -> None:
    """Appelé par Airflow à chaque fin de tâche réussie.

    Extrait records_count depuis le XCom de la tâche (HttpOperator stocke
    la réponse JSON en XCom). Les tâches utilitaires (resolve_*, notify_*)
    sont ignorées — elles n'ont pas de réponse API à afficher.
    """
    import json
    ti = context["task_instance"]
    if ti.task_id.startswith(("resolve_", "notify_")):
        return
    records = "N/A"
    raw = ti.xcom_pull(task_ids=ti.task_id)
    if raw:
        try:
            # .get() retourne None si la clé est présente mais vaut null (JSON null)
            # → on force "N/A" dans ce cas avec `or`
            records = json.loads(raw).get("records_count") or "N/A"
        except Exception:
            pass
    _notify_slack(f":white_check_mark: `{ti.task_id}` — records={records}")


def _on_star_schema_success(context) -> None:
    """Callback spécifique à load_star_schema.

    Extrait les champs détaillés de la réponse Gold (staging_rows, fact_rows,
    dimensions upsertées, snapshot_dt, elapsed) pour un message Slack plus riche
    que le callback générique _on_task_success (qui n'affiche que records_count).
    Gère aussi le cas skipped=True (snapshot déjà importé).
    """
    import json
    ti = context["task_instance"]
    raw = ti.xcom_pull(task_ids=ti.task_id)
    if not raw:
        _notify_slack(":white_check_mark: `load_star_schema` — pas de réponse disponible")
        return
    try:
        data = json.loads(raw)
    except Exception:
        _notify_slack(":white_check_mark: `load_star_schema` — réponse non parseable")
        return

    if data.get("skipped"):
        _notify_slack(
            ":white_check_mark: `load_star_schema` — aucun nouveau snapshot à importer"
        )
        return

    staging  = data.get("staging_rows")  or "?"
    fact     = data.get("fact_rows")     or "?"
    rome     = data.get("dim_code_rome_rows")     or "?"
    contrat  = data.get("dim_type_contrat_rows")  or "?"
    exp      = data.get("dim_experience_rows")    or "?"
    snap_dt  = data.get("snapshot_dt")   or "?"
    elapsed  = data.get("elapsed_sec")   or "?"

    _notify_slack(
        f":white_check_mark: `load_star_schema` — snapshot `{snap_dt}`\n"
        f">  staging → fact : *{staging}* → *{fact}* lignes\n"
        f">  dims  ROME={rome}  contrat={contrat}  expérience={exp}\n"
        f">  durée : {elapsed}s"
    )


def _on_dag_success(context) -> None:
    """Appelé par Airflow quand toutes les tâches du DAG run ont réussi."""
    dag_run = context["dag_run"]
    _notify_slack(
        f":tada: *Pipeline terminé* — DAG `{dag_run.dag_id}`\nrun : `{dag_run.run_id}`"
    )


# ---------------------------------------------------------------------------
# response_check partagé
# ---------------------------------------------------------------------------

def _check_response(response) -> bool:
    """Vérifie le succès de la réponse API et logue records_count.

    Tous les endpoints exposent records_count de façon homogène.
    Visible dans l'onglet Logs de chaque tâche dans l'UI Airflow.
    Retourne False → tâche marquée FAILED, retry déclenché.
    """
    data = response.json()
    success = data.get("success")
    records = data.get("records_count", "N/A")
    print(f"[jobmarket] success={success}  records_count={records}")
    return success is not False


# ---------------------------------------------------------------------------

# Paramètres appliqués à toutes les tâches du DAG par défaut.
default_args = {
    "owner": "jobmarket",
    "retries": 1,                              # 1 retry automatique en cas d'échec
    "retry_delay": timedelta(minutes=5),       # délai avant retry
    "on_failure_callback": _on_failure,        # Slack sur échec ou execution_timeout
    "on_success_callback": _on_task_success,   # Slack à chaque étape réussie (resolve_* et notify_* filtrés)
}

# Le bloc `with DAG(...) as dag:` déclare le DAG et toutes ses tâches.
# Toute tâche instanciée dans ce bloc est automatiquement rattachée au DAG.
with DAG(
    dag_id="jobmarket_daily",            # identifiant unique dans Airflow
    default_args=default_args,
    description="Pipeline quotidien : ingest → normalize → merge → status → gold",
    schedule="30 0 * * 1,3,5",          # lundi, mercredi, vendredi à 00h30
    start_date=datetime(2026, 1, 1),    # date à partir de laquelle le scheduling est actif
    catchup=False,                       # ne pas rejouer les runs manqués passés
    tags=["jobmarket"],                  # tag visible dans l'UI pour filtrer les DAGs
    on_success_callback=_on_dag_success, # Slack quand toutes les tâches ont réussi
    params={
        # Param expose ce paramètre dans l'UI lors d'un déclenchement manuel.
        # Type boolean → case à cocher dans "Trigger DAG w/ config".
        "debug": Param(
            False,
            type="boolean",
            description="Mode debug : limite FT à 2 codes ROME et WTTJ à 50 offres.",
        )
    },
) as dag:

    # --- Notification de démarrage ---
    #
    # Airflow n'a pas de callback natif de démarrage de DAG run.
    # On utilise un PythonOperator dédié comme première tâche.
    # Son préfixe "notify_" le fait ignorer par _on_task_success (pas de réponse API).

    def _notify_dag_start(**context):
        dag_run = context["dag_run"]
        _notify_slack(
            f":rocket: *Pipeline démarré* — DAG `{dag_run.dag_id}`\nrun : `{dag_run.run_id}`"
        )

    notify_start = PythonOperator(
        task_id="notify_start",
        python_callable=_notify_dag_start,
    )

    # --- Résolution dynamique des endpoints ---
    #
    # HttpOperator ne peut pas lire les params Airflow directement dans son URL.
    # On utilise deux PythonOperator intermédiaires qui :
    #   1. Lisent le param `debug` depuis le contexte d'exécution
    #   2. Retournent l'URL correcte
    #   3. Stockent cette URL dans XCom (mécanisme Airflow de passage de valeurs entre tâches)
    # Les HttpOperator suivants récupèrent l'URL via `xcom_pull`.

    def _build_ft_endpoint(**context):
        debug = context["params"].get("debug", False)
        if debug:
            return f"{API_INGEST_FT}?background=false&max_rome_codes={DEBUG_FT_MAX_ROME_CODES}"
        return f"{API_INGEST_FT}?background=false"

    def _build_wttj_endpoint(**context):
        debug = context["params"].get("debug", False)
        if debug:
            return f"{API_INGEST_WTTJ}?background=false&max_jobs={DEBUG_WTTJ_MAX_JOBS}&part_size={DEBUG_WTTJ_PART_SIZE}"
        return f"{API_INGEST_WTTJ}?background=false"

    resolve_ft_endpoint = PythonOperator(
        task_id="resolve_ft_endpoint",
        python_callable=_build_ft_endpoint,
    )

    resolve_wttj_endpoint = PythonOperator(
        task_id="resolve_wttj_endpoint",
        python_callable=_build_wttj_endpoint,
    )

    # --- Ingestion Bronze ---
    #
    # execution_timeout : durée maximale de la tâche elle-même (depuis son démarrage).
    # Si dépassée → AirflowTaskTimeout → on_failure_callback déclenché → Slack.
    # Distinct du SLA Airflow qui lui est mesuré depuis le démarrage du DAG run.

    ingest_ft = HttpOperator(
        task_id="ingest_ft",
        http_conn_id="jobmarket_api",    # connexion déclarée dans Airflow (host: jobmarket-api:8000)
        # xcom_pull récupère l'URL retournée par resolve_ft_endpoint
        endpoint="{{ task_instance.xcom_pull(task_ids='resolve_ft_endpoint') }}",
        method="POST",
        headers={"Content-Type": "application/json"},
        response_check=_check_response,
        log_response=True,              # log la réponse JSON dans les logs de la tâche
        execution_timeout=timedelta(hours=2),
    )

    ingest_wttj = HttpOperator(
        task_id="ingest_wttj",
        http_conn_id="jobmarket_api",
        endpoint="{{ task_instance.xcom_pull(task_ids='resolve_wttj_endpoint') }}",
        method="POST",
        headers={"Content-Type": "application/json"},
        response_check=_check_response,
        log_response=True,
        execution_timeout=timedelta(hours=12),
    )

    # --- Normalisation Silver (parallèle) ---

    normalize_wttj = HttpOperator(
        task_id="normalize_wttj",
        http_conn_id="jobmarket_api",
        endpoint=API_NORMALIZE_WTTJ,
        method="POST",
        headers={"Content-Type": "application/json"},
        response_check=_check_response,
        log_response=True,
        execution_timeout=timedelta(hours=1),
    )

    normalize_ft = HttpOperator(
        task_id="normalize_ft",
        http_conn_id="jobmarket_api",
        endpoint=API_NORMALIZE_FT,
        method="POST",
        headers={"Content-Type": "application/json"},
        response_check=_check_response,
        log_response=True,
        execution_timeout=timedelta(hours=1),
    )

    # --- Merge Silver ---

    merge = HttpOperator(
        task_id="merge",
        http_conn_id="jobmarket_api",
        endpoint=API_MERGE,
        method="POST",
        headers={"Content-Type": "application/json"},
        response_check=_check_response,
        log_response=True,
        execution_timeout=timedelta(minutes=30),
    )

    # --- Tracking du cycle de vie des offres ---

    status_tracking = HttpOperator(
        task_id="status_tracking",
        http_conn_id="jobmarket_api",
        endpoint=API_STATUS_TRACKING,
        method="POST",
        headers={"Content-Type": "application/json"},
        response_check=_check_response,
        log_response=True,
        execution_timeout=timedelta(minutes=30),
    )

    # --- Génération des datasets analytiques Silver ---

    status_evolution = HttpOperator(
        task_id="status_evolution",
        http_conn_id="jobmarket_api",
        endpoint=API_STATUS_EVOLUTION,
        method="POST",
        headers={"Content-Type": "application/json"},
        response_check=_check_response,
        log_response=True,
        execution_timeout=timedelta(minutes=30),
    )

    # --- Chargement Gold (star schema PostgreSQL) ---

    load_star_schema = HttpOperator(
        task_id="load_star_schema",
        http_conn_id="jobmarket_api",
        endpoint=API_LOAD_STAR_SCHEMA,
        method="POST",
        headers={"Content-Type": "application/json"},
        response_check=_check_response,
        log_response=True,
        execution_timeout=timedelta(minutes=30),
        on_success_callback=_on_star_schema_success,  # surcharge le callback générique
    )

    # --- Dépendances (ordre d'exécution) ---
    #
    # `>>` définit une dépendance : A >> B signifie "B s'exécute après A".
    # `[A, B]` signifie que les deux tâches s'exécutent en parallèle.
    #
    # resolve_* → ingest_* : les endpoints sont résolus avant l'appel HTTP
    # pour constuire l'URL avec ou sans paramètres de debug selon le cas
    notify_start >> [resolve_ft_endpoint, resolve_wttj_endpoint]
    resolve_ft_endpoint >> ingest_ft
    resolve_wttj_endpoint >> ingest_wttj
    #
    # ingest_ft → ingest_wttj : séquentiel (FT d'abord, plus rapide)
    # ingest_wttj → [normalize_wttj, normalize_ft] : parallèle une fois les deux ingest terminés
    # merge → status_tracking → status_evolution → load_star_schema : séquentiel
    ingest_ft >> ingest_wttj >> Label("Bronze → Silver") >> [normalize_wttj, normalize_ft] >> merge >> status_tracking >> status_evolution >> Label("Silver → Gold") >> load_star_schema
