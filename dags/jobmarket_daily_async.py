"""
DAG quotidien — Pipeline jobmarket complet (mode asynchrone).

Variante asynchrone de jobmarket_daily :
- Chaque tâche déclenche l'endpoint API avec background=true → réponse immédiate (task_id)
- Un polling sur GET /tasks/{task_id} attend la fin réelle du traitement
- Le résultat final est stocké en XCom (même format que la version sync)
  → callbacks Slack compatibles sans modification

Avantages vs version synchrone :
- Airflow ne maintient pas une connexion HTTP ouverte pendant des heures
- Progress visible dans les logs Airflow à chaque cycle de polling
- Timeout Airflow = filet de sécurité, pas une contrainte HTTP

Chaîne d'exécution :
    ingest_ft → ingest_wttj → [normalize_wttj, normalize_ft] → merge
    → status_tracking → status_evolution → load_star_schema

Mode debug :
    Déclencher via "Trigger DAG w/ config" → {"debug": true}
    → FT limité à 2 codes ROME, WTTJ limité à 50 offres.
"""

import json
import time
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.utils.edgemodifier import Label

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEBUG_FT_MAX_ROME_CODES  = 2
DEBUG_WTTJ_MAX_JOBS      = 50
DEBUG_WTTJ_PART_SIZE     = 50

API_INGEST_FT       = "/ingest/france-travail-offers"
API_INGEST_WTTJ     = "/ingest/welcome-to-jungle"
API_NORMALIZE_WTTJ  = "/data/normalize-wttj-jobs"
API_NORMALIZE_FT    = "/data/normalize-ft-jobs"
API_MERGE           = "/data/merge-datasets"
API_STATUS_TRACKING = "/data/status-tracking"
API_STATUS_EVOLUTION= "/data/status-evolution"
API_LOAD_STAR_SCHEMA= "/gold/load-star-schema?source_mode=auto"

# Intervalle de polling en secondes entre chaque vérification de statut
POLL_INTERVAL_FAST  = 30   # normalize, merge, status (tâches rapides)
POLL_INTERVAL_SLOW  = 60   # ingest FT / WTTJ (tâches longues)

# Statuts terminaux renvoyés par GET /tasks/{task_id}
_STATUS_SUCCESS = {"SUCCESS", "success", "completed"}
_STATUS_FAILED  = {"FAILED", "failed", "ERROR", "error"}

# ---------------------------------------------------------------------------
# Callbacks Slack (identiques à la version sync)
# ---------------------------------------------------------------------------

def _notify_slack(text: str) -> None:
    """Envoie un message Slack. Silent si SLACK_WEBHOOK_URL absent."""
    import requests as _requests
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    _requests.post(url, json={"text": text}, timeout=10)


def _on_failure(context) -> None:
    ti = context["task_instance"]
    exc = context.get("exception", "")
    _notify_slack(f":x: *Échec* — DAG `{ti.dag_id}` — tâche `{ti.task_id}`\n```{exc}```")


def _on_task_success(context) -> None:
    ti = context["task_instance"]
    if ti.task_id.startswith(("resolve_", "notify_")):
        return
    records = "N/A"
    raw = ti.xcom_pull(task_ids=ti.task_id)
    if raw:
        try:
            records = json.loads(raw).get("records_count") or "N/A"
        except Exception:
            pass
    _notify_slack(f":white_check_mark: `{ti.task_id}` — records={records}")


def _on_star_schema_success(context) -> None:
    """Callback enrichi pour load_star_schema (staging/fact/dims/elapsed)."""
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
        _notify_slack(":white_check_mark: `load_star_schema` — aucun nouveau snapshot à importer")
        return

    staging = data.get("staging_rows") or "?"
    fact    = data.get("fact_rows")    or "?"
    rome    = data.get("dim_code_rome_rows")    or "?"
    contrat = data.get("dim_type_contrat_rows") or "?"
    exp     = data.get("dim_experience_rows")   or "?"
    snap_dt = data.get("snapshot_dt")  or "?"
    elapsed = data.get("elapsed_sec")  or "?"

    _notify_slack(
        f":white_check_mark: `load_star_schema` — snapshot `{snap_dt}`\n"
        f">  staging → fact : *{staging}* → *{fact}* lignes\n"
        f">  dims  ROME={rome}  contrat={contrat}  expérience={exp}\n"
        f">  durée : {elapsed}s"
    )


def _on_dag_success(context) -> None:
    dag_run = context["dag_run"]
    _notify_slack(f":tada: *Pipeline terminé* — DAG `{dag_run.dag_id}`\nrun : `{dag_run.run_id}`")


# ---------------------------------------------------------------------------
# Moteur async — cœur de la version asynchrone
# ---------------------------------------------------------------------------

def _async_call(endpoint: str, poll_interval: int = POLL_INTERVAL_FAST, **context) -> str:
    """Déclenche un endpoint API en background, poll jusqu'à completion.

    Séquence :
        1. POST {endpoint}?background=true → { task_id / job_id / run_id }
        2. GET /tasks/{task_id} toutes les {poll_interval}s → progress_pct, status
        3. Retourne le résultat final sérialisé en JSON (stocké dans XCom)

    Levées d'exception :
        - ValueError : pas de task_id dans la réponse de déclenchement
        - RuntimeError : tâche terminée en FAILED côté API
        - Les timeouts Airflow (execution_timeout) interrompent la boucle de polling

    XCom retourné :
        JSON contenant au minimum success, records_count (et les champs spécifiques
        à l'endpoint) — format identique aux réponses sync pour compatibilité callbacks.
    """
    import requests
    from airflow.hooks.base import BaseHook

    # Lire l'URL de base depuis la connexion Airflow "jobmarket_api"
    # (même source que HttpOperator dans la version sync)
    conn = BaseHook.get_connection("jobmarket_api")
    base_url = f"{conn.conn_type or 'http'}://{conn.host}:{conn.port}"

    trigger_url = f"{base_url}{endpoint}"
    # Forcer background=true — ajouter au query string existant
    sep = "&" if "?" in trigger_url else "?"
    trigger_url = f"{trigger_url}{sep}background=true"

    # 1. Déclencher la tâche en arrière-plan
    print(f"[async] POST {trigger_url}")
    resp = requests.post(trigger_url, headers={"Content-Type": "application/json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"[async] Réponse déclenchement : {json.dumps(data)}")

    # Récupérer l'identifiant de tâche depuis le champ task_id (présent sur tous les
    # retours background depuis BaseJobResponse).
    # Fallback run_id : endpoints d'ingestion (FT, WTTJ) trackés dans JobStore
    # → polled via GET /jobs/{id} au lieu de GET /tasks/{id}.
    task_id = data.get("task_id")
    if not task_id:
        raise ValueError(f"Pas de task_id dans la réponse : {data}")

    if data.get("run_id"):
        poll_url = f"{base_url}/jobs/{task_id}"
    else:
        poll_url = f"{base_url}/tasks/{task_id}"

    print(f"[async] Tâche démarrée : {task_id} — polling {poll_url} toutes les {poll_interval}s")

    # 2. Boucle de polling
    iteration = 0
    while True:
        time.sleep(poll_interval)
        iteration += 1

        try:
            poll_resp = requests.get(poll_url, timeout=15)
            poll_resp.raise_for_status()
            state = poll_resp.json()
        except Exception as e:
            # Erreur réseau transitoire — on continue le polling
            print(f"[async] Erreur polling #{iteration} : {e} — retry dans {poll_interval}s")
            continue

        if state is None:
            print(f"[async] #{iteration} réponse vide — retry dans {poll_interval}s")
            continue

        status      = state.get("status", "unknown")
        progress    = state.get("progress_pct", "?")
        message     = state.get("message", "")
        records     = state.get("records_count") or (state.get("result") or {}).get("records_count", "?")

        print(f"[async] #{iteration} | status={status} | progress={progress}% | records={records} | {message}")

        if status in _STATUS_SUCCESS:
            # Construire le payload XCom à partir du résultat final
            # On fusionne les champs du state + result pour couvrir tous les cas
            result_payload = {
                "success": True,
                "message": message,
                "records_count": state.get("records_count"),
                **(state.get("result") or {}),  # staging_rows, fact_rows, snapshot_dt, etc.
            }
            print(f"[async] Terminé avec succès : {json.dumps(result_payload)}")
            return json.dumps(result_payload)

        if status in _STATUS_FAILED:
            error = state.get("error") or message
            raise RuntimeError(f"Tâche {task_id} échouée côté API : {error}")

        # Statut intermédiaire (RUNNING, pending, etc.) → continuer


# ---------------------------------------------------------------------------
# default_args
# ---------------------------------------------------------------------------

default_args = {
    "owner": "jobmarket",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": _on_failure,
    "on_success_callback": _on_task_success,
}

# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="jobmarket_daily_async",
    default_args=default_args,
    description="Pipeline quotidien (mode async) : ingest → normalize → merge → status → gold",
    schedule="30 0 * * 1,3,5",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["jobmarket", "async"],
    on_success_callback=_on_dag_success,
    params={
        "debug": Param(
            False,
            type="boolean",
            description="Mode debug : limite FT à 2 codes ROME et WTTJ à 50 offres.",
        )
    },
) as dag:

    # --- Notification démarrage ---

    def _notify_dag_start(**context):
        dag_run = context["dag_run"]
        _notify_slack(f":rocket: *Pipeline démarré (async)* — DAG `{dag_run.dag_id}`\nrun : `{dag_run.run_id}`")

    notify_start = PythonOperator(
        task_id="notify_start",
        python_callable=_notify_dag_start,
    )

    # --- Ingestion FT (async) ---
    #
    # L'endpoint est construit dynamiquement pour le mode debug.
    # _async_call déclenche POST ?background=true puis poll jusqu'à SUCCESS.

    def _ingest_ft(**context):
        debug = context["params"].get("debug", False)
        endpoint = API_INGEST_FT
        if debug:
            endpoint += f"?max_rome_codes={DEBUG_FT_MAX_ROME_CODES}"
        return _async_call(endpoint, poll_interval=POLL_INTERVAL_SLOW, **context)

    ingest_ft = PythonOperator(
        task_id="ingest_ft",
        python_callable=_ingest_ft,
        execution_timeout=timedelta(hours=3),
    )

    # --- Ingestion WTTJ (async) ---

    def _ingest_wttj(**context):
        debug = context["params"].get("debug", False)
        endpoint = API_INGEST_WTTJ
        if debug:
            endpoint += f"?max_jobs={DEBUG_WTTJ_MAX_JOBS}&part_size={DEBUG_WTTJ_PART_SIZE}"
        return _async_call(endpoint, poll_interval=POLL_INTERVAL_SLOW, **context)

    ingest_wttj = PythonOperator(
        task_id="ingest_wttj",
        python_callable=_ingest_wttj,
        execution_timeout=timedelta(hours=13),
    )

    # --- Normalisation Silver (parallèle) ---

    def _normalize_wttj(**context):
        return _async_call(API_NORMALIZE_WTTJ, poll_interval=POLL_INTERVAL_FAST, **context)

    def _normalize_ft(**context):
        return _async_call(API_NORMALIZE_FT, poll_interval=POLL_INTERVAL_FAST, **context)

    normalize_wttj = PythonOperator(
        task_id="normalize_wttj",
        python_callable=_normalize_wttj,
        execution_timeout=timedelta(hours=2),
    )

    normalize_ft = PythonOperator(
        task_id="normalize_ft",
        python_callable=_normalize_ft,
        execution_timeout=timedelta(hours=2),
    )

    # --- Merge Silver ---

    def _merge(**context):
        return _async_call(API_MERGE, poll_interval=POLL_INTERVAL_FAST, **context)

    merge = PythonOperator(
        task_id="merge",
        python_callable=_merge,
        execution_timeout=timedelta(minutes=45),
    )

    # --- Status tracking ---

    def _status_tracking(**context):
        return _async_call(API_STATUS_TRACKING, poll_interval=POLL_INTERVAL_FAST, **context)

    status_tracking = PythonOperator(
        task_id="status_tracking",
        python_callable=_status_tracking,
        execution_timeout=timedelta(minutes=45),
    )

    # --- Status evolution ---

    def _status_evolution(**context):
        return _async_call(API_STATUS_EVOLUTION, poll_interval=POLL_INTERVAL_FAST, **context)

    status_evolution = PythonOperator(
        task_id="status_evolution",
        python_callable=_status_evolution,
        execution_timeout=timedelta(minutes=45),
    )

    # --- Load star schema Gold ---

    def _load_star_schema(**context):
        return _async_call(API_LOAD_STAR_SCHEMA, poll_interval=POLL_INTERVAL_FAST, **context)

    load_star_schema = PythonOperator(
        task_id="load_star_schema",
        python_callable=_load_star_schema,
        execution_timeout=timedelta(minutes=45),
        on_success_callback=_on_star_schema_success,
    )

    # --- Dépendances ---

    notify_start >> [ingest_ft, ingest_wttj]
    ingest_ft >> ingest_wttj
    ingest_wttj >> Label("Bronze → Silver") >> [normalize_wttj, normalize_ft]
    [normalize_wttj, normalize_ft] >> merge
    merge >> status_tracking >> status_evolution >> Label("Silver → Gold") >> load_star_schema
