""" 
Main API module for ROME classification and data ingestion.

This module defines the FastAPI application, endpoints for prediction 
and ingestion, and integrates with the JobStore for tracking long-running tasks.

It also uses log_to_db utility to log ingestion events directly to PostgreSQL,
with graceful degradation if psycopg is not installed.
"""
import os
from src.config.env import require_env, get_project_root, load_project_env
load_project_env()  # safe à rappeler (idempotent)

ENABLE_GRAFANA_LOGS = os.getenv("ENABLE_GRAFANA_LOGS", "false").lower() == "true"
JOBSTORE_DSN = os.getenv("JOBSTORE_DSN")
STALE_JOB_MINUTES = int(os.getenv("STALE_JOB_MINUTES", "15"))

STATUS_RUNNING = "RUNNING"
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"

MODEL_NAME = os.getenv("MODEL_NAME", "rome_tfidf")
TOP_K = int(os.getenv("TOP_K", "5"))

import uuid
import time
import pandas as pd
import logging
import json
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

# Model
from src.models.predict_model import build_text_payload, load_artifacts, predict_top_k, get_rome_model

# Imports applicatifs
from src.ingest.silver.normalize_wttj_jobs import normalize_wttj_jobs
from src.ingest.silver.normalize_ft_jobs import normalize_ft_jobs
from src.ingest.bronze.ingest_france_travail_rome_metiers import ingest_rome_metiers
from src.ingest.bronze.ingest_france_travail_jobs import ingest_france_travail_offers
from src.ingest.bronze.ingest_wttj_jobs import ingest_welcome_to_the_jungle
from src.ingest.bronze.ingest_wttj_collect_urls import collect_sitemap_urls
from src.ingest.bronze.ingest_wttj_job_opt import ingest_welcome_to_the_jungle_opt
from src.ingest.silver.merge_ft_wttj_datasets import merge_ft_wttj_datasets
from src.observability.job_store import JobStore
from src.utils.log_to_db import log_to_db
from src.utils.time_helpers import format_eta, utc_run_id

from src.api.models import (
    PredictRequest,
    PredictResponse,
    IngestResponse,
    IngestOffersResponse,
    IngestWTTJResponse,
    CollectSitemapsResponse,
    IngestWTTJOptResponse,
    MergeDatasetResponse,
    NormalizeWTTJResponse,
    NormalizeFTResponse,
    JSONOnlyFilter
)

# Global cache (loaded once at startup)
ARTIFACTS: Dict[str, Any] = {}
rome_model = None

# Tracking tasks in progress 
ACTIVE_TASKS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="ROME Classifier & Data Ingestion API",
    version="1.0.0",
    description="""
API microservice pour la classification automatique des offres d'emploi selon le référentiel ROME 
et l'ingestion des données de référence depuis France Travail.

## Fonctionnalités

- **Prédiction ROME** : Classifie une offre d'emploi et retourne les codes ROME correspondants
- **Ingestion des codes ROME** : Importe la nomenclature complète des métiers ROME
- **Ingestion des offres d'emploi** : Importe les offres d'emploi depuis France Travail
- **Monitoring** : Status et health checks de l'API

## Utilisation

Consultez les endpoints ci-dessous pour tester l'API directement depuis cette interface.
    """,
    contact={
        "name": "Job Market API Support"
    }
)

def normalize_wttj_jobs_task(dt: str, output_format: str = "parquet"):
    result = normalize_wttj_jobs(dt, output_format)
    return NormalizeWTTJResponse(
        job_id=result.job_id,
        status=result.status,
        dt=result.dt,
        format=result.format,
        files=result.files,
        errors=result.errors
    )


def normalize_ft_jobs_task(dt: str, output_format: str = "parquet"):
    result = normalize_ft_jobs(dt, output_format)
    return NormalizeFTResponse(
        job_id=result.job_id,
        status=result.status,
        dt=result.dt,
        format=result.format,
        files=result.files,
        errors=result.errors,
    )


def run_normalize_wttj_task(task_id: str, dt: Optional[str], output_format: str):
    """Wrapper for WTTJ normalization with task tracking."""
    start_monotonic = time.monotonic()
    try:
        result = normalize_wttj_jobs_task(dt, output_format)
        duration_sec = time.monotonic() - start_monotonic
        if result.status == STATUS_SUCCESS:
            result_payload = {
                "dt": result.dt,
                "format": result.format,
                "files": result.files,
                "errors": result.errors,
                "duration_sec": round(duration_sec, 2),
            }
            set_task(
                task_id,
                progress_pct=100,
                message="Normalisation WTTJ terminee",
                records_count=len(result.files),
                errors_count=result.errors,
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )
            if job_store.enabled:
                job_store.finish(task_id, STATUS_SUCCESS, result=result_payload)
        else:
            error_text = f"Echec normalisation WTTJ: {result.status}"
            set_task(
                task_id,
                message=error_text,
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                errors_count=result.errors,
                error=error_text,
            )
            if job_store.enabled:
                job_store.finish(task_id, STATUS_FAILED, error_text=error_text)
    except Exception as e:
        error_text = str(e)
        set_task(
            task_id,
            message=f"Erreur: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        if job_store.enabled:
            job_store.finish(task_id, STATUS_FAILED, error_text=error_text)


def run_normalize_ft_task(task_id: str, dt: Optional[str], output_format: str):
    """Wrapper for FT normalization with task tracking."""
    start_monotonic = time.monotonic()
    try:
        result = normalize_ft_jobs_task(dt, output_format)
        duration_sec = time.monotonic() - start_monotonic
        if result.status == STATUS_SUCCESS:
            result_payload = {
                "dt": result.dt,
                "format": result.format,
                "files": result.files,
                "errors": result.errors,
                "duration_sec": round(duration_sec, 2),
            }
            set_task(
                task_id,
                progress_pct=100,
                message="Normalisation FT terminee",
                records_count=len(result.files),
                errors_count=result.errors,
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )
            if job_store.enabled:
                job_store.finish(task_id, STATUS_SUCCESS, result=result_payload)
        else:
            error_text = f"Echec normalisation FT: {result.status}"
            set_task(
                task_id,
                message=error_text,
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                errors_count=result.errors,
                error=error_text,
            )
            if job_store.enabled:
                job_store.finish(task_id, STATUS_FAILED, error_text=error_text)
    except Exception as e:
        error_text = str(e)
        set_task(
            task_id,
            message=f"Erreur: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        if job_store.enabled:
            job_store.finish(task_id, STATUS_FAILED, error_text=error_text)

@app.post(
    "/data/normalize-wttj-jobs",
    response_model=NormalizeWTTJResponse,
    tags=["Data Processing"],
    summary="Normalise les jobs WTTJ bronze en silver avec code ROME",
    description="Lit tous les jobs_raw du bronze pour un dt donné, prédit le code ROME, écrit le résultat dans la couche silver en gardant l’arborescence, format choisi."
)
async def normalize_wttj_jobs_endpoint(
    background_tasks: BackgroundTasks,
    dt: Optional[str] = Query(None, description="Date time d'extraction dans la couche bronze au format YYYY-MM-DD (ex: 2026-02-28). Si non fourni ou 'latest', prend le dernier dt disponible dans le storage."),
    output_format: str = Query(default="parquet", description="Format de sortie: parquet (par défaut), jsonl, ou csv"),
    background: bool = Query(default=True, description="Exécuter la tâche en arrière-plan")
):
    if background:
        task_id = utc_run_id()
        ACTIVE_TASKS[task_id] = {
            "operation": "normalize_wttj_jobs",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Normalisation WTTJ en cours (format: {output_format})...",
            "params": {"dt": dt, "output_format": output_format},
        }
        if job_store.enabled:
            job_store.create(
                run_id=task_id,
                job_type="data",
                source="normalize_wttj_jobs",
                params={"background": True, "dt": dt, "output_format": output_format},
                message=f"Normalisation WTTJ en cours (format: {output_format})...",
            )
        background_tasks.add_task(run_normalize_wttj_task, task_id, dt, output_format)
        return NormalizeWTTJResponse(job_id=task_id, status="RUNNING", dt=dt, format=output_format, files=[], errors=0)
    else:
        return normalize_wttj_jobs_task(dt, output_format)


@app.post(
    "/data/normalize-ft-jobs",
    response_model=NormalizeFTResponse,
    tags=["Data Processing"],
    summary="Normalise les jobs FT bronze en silver",
    description="Lit tous les jobs_raw FT du bronze pour un dt donne, applique la normalisation source et ecrit le resultat canonique dans la couche silver."
)
async def normalize_ft_jobs_endpoint(
    background_tasks: BackgroundTasks,
    dt: Optional[str] = Query(None, description="Date d'extraction dans la couche bronze au format YYYY-MM-DD (ex: 2026-02-28). Si non fourni ou 'latest', prend le dernier dt disponible dans le storage."),
    output_format: str = Query(default="parquet", description="Format de sortie: parquet (par defaut), jsonl, ou csv"),
    background: bool = Query(default=True, description="Executer la tache en arriere-plan")
):
    if background:
        task_id = utc_run_id()
        ACTIVE_TASKS[task_id] = {
            "operation": "normalize_ft_jobs",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Normalisation FT en cours (format: {output_format})...",
            "params": {"dt": dt, "output_format": output_format},
        }
        if job_store.enabled:
            job_store.create(
                run_id=task_id,
                job_type="data",
                source="normalize_ft_jobs",
                params={"background": True, "dt": dt, "output_format": output_format},
                message=f"Normalisation FT en cours (format: {output_format})...",
            )
        background_tasks.add_task(run_normalize_ft_task, task_id, dt, output_format)
        return NormalizeFTResponse(job_id=task_id, status="RUNNING", dt=dt, format=output_format, files=[], errors=0)
    else:
        return normalize_ft_jobs_task(dt, output_format)

# Configure logging with rotation and optional Grafana support
def setup_logging():
    """Configure logging with console and rotating file handlers, plus optional JSON logging for Grafana"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10*1024*1024))  # 10MB by default
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Create log directories
    Path("logs/api").mkdir(parents=True, exist_ok=True)
    Path("logs/ingestion").mkdir(parents=True, exist_ok=True)
    Path("logs/prediction").mkdir(parents=True, exist_ok=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Standard formatter (readable for humans)
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Base handlers (console + rotating files)
    if not root_logger.handlers:
        # 1. Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(standard_formatter)
        console_handler.setLevel(getattr(logging, log_level))

        # 2. Main API file
        file_handler = RotatingFileHandler(
            'logs/api/main.log',
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(standard_formatter)
        file_handler.setLevel(getattr(logging, log_level))

        # 3. Global error file
        error_handler = RotatingFileHandler(
            'logs/api/errors.log',
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding='utf-8'
        )
        error_handler.setFormatter(standard_formatter)
        error_handler.setLevel(logging.ERROR)

        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)

        # Propagate log level to uvicorn loggers
        for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uvicorn_logger = logging.getLogger(logger_name)
            uvicorn_logger.handlers = root_logger.handlers
            uvicorn_logger.setLevel(getattr(logging, log_level))
            uvicorn_logger.propagate = False
    
    # 4. Logger dedicated to structured events (JSON only)
    structured_logger = logging.getLogger("structured")
    structured_logger.setLevel(logging.INFO)
    structured_logger.propagate = False
    structured_logger.handlers.clear()

    if ENABLE_GRAFANA_LOGS:
        json_handler = logging.FileHandler('logs/api/structured.jsonl', encoding='utf-8')
        json_handler.setFormatter(logging.Formatter('%(message)s'))
        json_handler.setLevel(logging.INFO)
        json_handler.name = "json_handler"
        json_handler.addFilter(JSONOnlyFilter())  # JSON only filter
        structured_logger.addHandler(json_handler)
        root_logger.info("📊 Logs structurés Grafana activés")
    
    root_logger.info(f"📝 Logging configuré - Niveau: {log_level}, Grafana: {ENABLE_GRAFANA_LOGS}")
    
    return root_logger


# Fonction get_endpoint_logger() supprimée - remplacée par log_to_db()
# Les logs d'ingestion sont maintenant stockés directement en PostgreSQL (table ingestion_logs)
logger = setup_logging()
structured_logger = logging.getLogger("structured")
job_store = JobStore(JOBSTORE_DSN)


def emit_structured_log(payload: Dict[str, Any]) -> None:
    """Route un événement JSON vers le logger structuré si Grafana est activé."""
    if not ENABLE_GRAFANA_LOGS:
        return
    structured_logger.info(json.dumps(payload))

""" 
Set task status and metadata in ACTIVE_TASKS and JobStore (if enabled) 
Extra can include any additional info to track (e.g. current_rome, current_rome_label for offers ingestion)
"""

def set_task(
    task_id: str,
    progress_pct: Optional[int] = None,
    message: Optional[str] = None,
    records_count: Optional[int] = None,
    pages_count: Optional[int] = None,
    errors_count: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> None:
    if task_id in ACTIVE_TASKS:
        ACTIVE_TASKS[task_id].update(extra)
    else:
        ACTIVE_TASKS[task_id] = dict(extra)
    if progress_pct is not None:
        ACTIVE_TASKS[task_id]["progress_pct"] = progress_pct
    if message is not None:
        ACTIVE_TASKS[task_id]["message"] = message
    if job_store.enabled:
        job_store.progress(
            task_id,
            progress_pct=progress_pct,
            message=message,
            records_count=records_count,
            pages_count=pages_count,
            errors_count=errors_count,
            result=result,
        )

@app.on_event("startup")
def _startup_load_model():
    """
    Load model and artifacts at startup.
    Avoid reloading MinIO/joblib on each request.
    """
    global ARTIFACTS, rome_model
    logger.info("🚀 Démarrage de l'API - Chargement du modèle...")
    try:
        ARTIFACTS = load_artifacts()
        logger.info(f"✅ Modèle chargé avec succès: {MODEL_NAME} v{ARTIFACTS['version']}")
        rome_model = get_rome_model()
        
        # Log structuré si Grafana activé
        if ENABLE_GRAFANA_LOGS:
            structured_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "model_loaded",
                "model_name": MODEL_NAME,
                "version": ARTIFACTS['version']
            }
            emit_structured_log(structured_log)

        if job_store.enabled:
            stale_count = job_store.mark_stale(STALE_JOB_MINUTES)
            if stale_count:
                logger.warning("Marquage des jobs stale: %s", stale_count)
                    
    except Exception as e:
        ARTIFACTS = {}
        rome_model = {}
        logger.warning(
            "⚠️ Modèle indisponible au démarrage (API continue sans crash): %s",
            e,
        )


# =====================================
# Wrappers for ingest tasks with tracking
# =====================================

def run_rome_metiers_task(task_id: str):
    """Wrapper for ingesting ROME codes with status updates"""
    import time
    
    start_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)

    if ENABLE_GRAFANA_LOGS:
        emit_structured_log({
            "timestamp": started_at.isoformat(),
            "event_type": "job_started",
            "run_id": task_id,
            "source": "rome_metiers",
            "status": STATUS_RUNNING,
            "progress_pct": 0,
            "records_count": 0,
            "pages_count": 0,
            "errors_count": 0
        })

    try:
        log_to_db('rome_metiers', 'INFO', "📥 Début de l'ingestion des codes ROME", task_id=task_id)
        result = ingest_rome_metiers()

        duration_sec = time.monotonic() - start_monotonic

        if result["success"]:
            result_payload = {
                "records_count": result.get("records_count"),
                "records_written": result.get("records_written"),
                "key": result.get("key"),
                "duration_sec": round(duration_sec, 2)
            }
            set_task(
                task_id,
                progress_pct=100,
                message=result["message"],
                records_count=result.get("records_count"),
                pages_count=result.get("calls", 0),
                errors_count=result.get("errors", 0),
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )
            log_to_db(
                'rome_metiers',
                'INFO',
                f"✅ {result.get('records_written')} codes ROME métiers importés ({result.get('records_count')} total) - {duration_sec:.2f}s",
                task_id=task_id,
                duration_sec=round(duration_sec, 2),
                records_count=result.get('records_count'),
                records_written=result.get('records_written'),
                key=result.get('key')
            )
            if job_store.enabled:
                job_store.finish(task_id, STATUS_SUCCESS, result=result_payload)
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_finished",
                    "run_id": task_id,
                    "source": "rome_metiers",
                    "status": STATUS_SUCCESS,
                    "progress_pct": 100,
                    "records_count": result.get("records_count"),
                    "pages_count": result.get("calls", 0),
                    "errors_count": result.get("errors", 0),
                    "duration_sec": round(duration_sec, 2)
                })
        else:
            error_text = result.get("error")
            set_task(
                task_id,
                message=result.get("message", "Échec de l'ingestion"),
                records_count=result.get("records_count"),
                pages_count=result.get("calls", 0),
                errors_count=result.get("errors", 0),
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )
            log_to_db('rome_metiers', 'ERROR', f"❌ Échec de l'ingestion: {result.get('error')}", task_id=task_id, error=result.get('error'))
            if job_store.enabled:
                job_store.finish(task_id, STATUS_FAILED, result=result, error_text=error_text)
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_failed",
                    "run_id": task_id,
                    "source": "rome_metiers",
                    "status": STATUS_FAILED,
                    "progress_pct": None,
                    "records_count": result.get("records_count", 0),
                    "pages_count": result.get("calls", 0),
                    "errors_count": result.get("errors", 0),
                    "duration_sec": round(duration_sec, 2)
                })
                        
    except Exception as e:
        duration_sec = time.monotonic() - start_monotonic
        error_text = str(e)
        set_task(
            task_id,
            message=f"Erreur: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        log_to_db('rome_metiers', 'ERROR', f"❌ Exception après {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
        if job_store.enabled:
            job_store.finish(task_id, STATUS_FAILED, error_text=error_text)
        if ENABLE_GRAFANA_LOGS:
            emit_structured_log({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "job_failed",
                "run_id": task_id,
                "source": "rome_metiers",
                "status": STATUS_FAILED,
                "progress_pct": None,
                "records_count": 0,
                "pages_count": 0,
                "errors_count": 1,
                "duration_sec": round(duration_sec, 2)
            })


def run_france_travail_offers_task(task_id: str, window_days: int, max_windows: int, 
                                     binary_split_min_seconds: int, max_rome_codes: int):
    """Wrapper for ingesting France Travail offers with status updates and progress tracking"""
    import time
    
    start_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)

    if ENABLE_GRAFANA_LOGS:
        emit_structured_log({
            "timestamp": started_at.isoformat(),
            "event_type": "job_started",
            "run_id": task_id,
            "source": "france_travail_offers",
            "status": STATUS_RUNNING,
            "progress_pct": 0,
            "records_count": 0,
            "pages_count": 0,
            "errors_count": 0
        })

    def update_progress(current: int, total: int, rome_code: str, rome_label: str):
        """Callback to update progress in real-time"""
        progress_pct = int((current / total) * 100) if total else 0
        set_task(
            task_id,
            progress_pct=progress_pct,
            message=f"Traitement code ROME {current}/{total}: {rome_code} - {rome_label}",
            current_rome=rome_code,
            current_rome_label=rome_label,
        )
    
    try:
        log_to_db('france_travail_offers', 'INFO', "📥 Début de l'ingestion des offres France Travail", task_id=task_id)
        result = ingest_france_travail_offers(
            storage=None,
            client=None,
            window_days=window_days,
            max_windows=max_windows,
            binary_split_min_seconds=binary_split_min_seconds,
            max_rome_codes=max_rome_codes,
            progress_callback=update_progress,
            logger_override=None,
            task_id=task_id,
        )
        
        duration_sec = time.monotonic() - start_monotonic

        if result["success"]:
            result_payload = {
                "run_id": result.get("run_id"),
                "run_key": result.get("run_key"),
                "rome_processed": result.get("rome_processed"),
                "calls": result.get("calls"),
                "written": result.get("written"),
                "elapsed_s": result.get("elapsed_s"),
                "errors": result.get("errors"),
                "duration_sec": round(duration_sec, 2)
            }
            set_task(
                task_id,
                progress_pct=100,
                message=result["message"],
                records_count=result.get("written", 0),
                pages_count=result.get("calls", 0),
                errors_count=result.get("errors", 0),
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )
            log_to_db(
                'france_travail_offers',
                'INFO',
                (
                    f"✅ {result.get('written')} offres importées ({result.get('rome_processed')} codes ROME, "
                    f"{result.get('calls')} appels, {result.get('errors')} erreurs) - {format_eta(duration_sec)}"
                ),
                task_id=task_id,
                duration_sec=round(duration_sec, 2),
                records_count=result.get('written', 0),
                error_count=result.get('errors', 0),
                rome_processed=result.get('rome_processed'),
                api_calls=result.get('calls')
            )

            if job_store.enabled:
                job_store.finish(task_id, STATUS_SUCCESS, result=result_payload)
            
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_finished",
                    "run_id": task_id,
                    "source": "france_travail_offers",
                    "status": STATUS_SUCCESS,
                    "progress_pct": 100,
                    "records_count": result.get("written", 0),
                    "pages_count": result.get("calls", 0),
                    "errors_count": result.get("errors", 0),
                    "duration_sec": round(duration_sec, 2)
                })
        else:
            error_text = result.get("error")
            
            set_task(
                task_id,
                message=result.get("message", "Échec de l'ingestion"),
                records_count=result.get("written", 0),
                pages_count=result.get("calls", 0),
                errors_count=result.get("errors", 0),
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )
            
            log_to_db('france_travail_offers', 'ERROR', f"❌ Échec de l'ingestion: {result.get('error')}", task_id=task_id, error=result.get('error'))
            
            if job_store.enabled:
                job_store.finish(task_id, STATUS_FAILED, result=result, error_text=error_text)
            
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_failed",
                    "run_id": task_id,
                    "source": "france_travail_offers",
                    "status": STATUS_FAILED,
                    "progress_pct": None,
                    "records_count": result.get("written", 0),
                    "pages_count": result.get("calls", 0),
                    "errors_count": result.get("errors", 0),
                    "duration_sec": round(duration_sec, 2)
                })

    except Exception as e:
        duration_sec = time.monotonic() - start_monotonic
        error_text = str(e)
        set_task(
            task_id,
            message=f"Erreur: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )

        log_to_db('france_travail_offers', 'ERROR', f"❌ Exception après {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
        
        if job_store.enabled:
            job_store.finish(task_id, STATUS_FAILED, error_text=error_text)
        
        if ENABLE_GRAFANA_LOGS:
            emit_structured_log({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "job_failed",
                "run_id": task_id,
                "source": "france_travail_offers",
                "status": STATUS_FAILED,
                "progress_pct": None,
                "records_count": 0,
                "pages_count": 0,
                "errors_count": 1,
                "duration_sec": round(duration_sec, 2)
            })


def run_welcome_to_jungle_task(
    task_id: str,
    mode: str,
    max_jobs: int,
    max_companies: int,
    workers: int,
    part_size: int,
    provided_run_id: str | None,
    resume_from_run_id: str | None,
):
    """Wrapper for ingesting Welcome to the Jungle data with status updates"""
    import time

    start_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)

    if ENABLE_GRAFANA_LOGS:

        emit_structured_log({
            "timestamp": started_at.isoformat(),
            "event_type": "job_started",
            "run_id": task_id,
            "source": "wttj",
            "status": STATUS_RUNNING,
            "progress_pct": 0,
            "records_count": 0,
            "pages_count": 0,
            "errors_count": 0
        })
    
    def update_progress(segment: str, current: int, total: int, ok: int, ko: int):
        """Callback to update progress in real-time"""
        progress_pct = int((current / total) * 100) if total > 0 else 0
        set_task(
            task_id,
            progress_pct=progress_pct,
            message=f"Segment {segment}: {current}/{total} URLs traitées (✓{ok} ✗{ko})",
            pages_count=current,
            errors_count=ko,
            current_segment=segment,
            current_url=current,
            total_urls=total,
        )
    
    try:
        log_to_db('welcome_to_the_jungle', 'INFO', "📥 Début de l'ingestion Welcome to the Jungle", task_id=task_id)
        
        result = ingest_welcome_to_the_jungle(
            storage=None,
            mode=mode,
            max_jobs=max_jobs,
            max_companies=max_companies,
            workers=workers,
            part_size=part_size,
            provided_run_id=provided_run_id,
            resume_from_run_id=resume_from_run_id,
            progress_callback=update_progress,
        )
        
        duration_sec = time.monotonic() - start_monotonic
        
        if result["success"]:
            result_payload = {
                "run_id": result.get("run_id"),
                "dt": result.get("dt"),
                "mode": result.get("mode"),
                "total_processed": result.get("total_processed"),
                "total_written": result.get("total_written"),
                "elapsed_s": result.get("elapsed_s"),
                "jobs": result.get("jobs"),
                "companies": result.get("companies")
            }
            errors_count = 0
            for segment in (result.get("jobs"), result.get("companies")):
                if isinstance(segment, dict):
                    errors_count += segment.get("errors", 0)

            set_task(
                task_id,
                progress_pct=100,
                message=result["message"],
                records_count=result.get("total_written", 0),
                pages_count=result.get("total_processed", 0),
                errors_count=errors_count,
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )

            log_to_db(
                'welcome_to_the_jungle',
                'INFO',
                (
                    f"✅ {result.get('total_written')} offres importées ({result.get('total_processed')} URLs, "
                    f"{errors_count} erreurs) - {format_eta(duration_sec)}"
                ),
                task_id=task_id,
                duration_sec=round(duration_sec, 2),
                records_count=result.get('total_written', 0),
                error_count=errors_count,
                urls_processed=result.get('total_processed')
            )
            
            if job_store.enabled:
                job_store.finish(task_id, STATUS_SUCCESS, result=result_payload)
            
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_finished",
                    "run_id": task_id,
                    "source": "wttj",
                    "status": STATUS_SUCCESS,
                    "progress_pct": 100,
                    "records_count": result.get("total_written", 0),
                    "pages_count": result.get("total_processed", 0),
                    "errors_count": errors_count,
                    "duration_sec": round(duration_sec, 2)
                })
        else:
            error_text = result.get("error")
            set_task(
                task_id,
                message=result.get("message", "Échec de l'ingestion"),
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )

            log_to_db('welcome_to_the_jungle', 'ERROR', f"❌ Échec de l'ingestion: {result.get('error')}", task_id=task_id, error=result.get('error'))
            
            if job_store.enabled:
                job_store.finish(task_id, STATUS_FAILED, result=result, error_text=error_text)
            
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_failed",
                    "run_id": task_id,
                    "source": "wttj",
                    "status": STATUS_FAILED,
                    "progress_pct": None,
                    "records_count": result.get("total_written", 0),
                    "pages_count": result.get("total_processed", 0),
                    "errors_count": 1,
                    "duration_sec": round(duration_sec, 2)
                })

    except Exception as e:
        duration_sec = time.monotonic() - start_monotonic
        error_text = str(e)
        
        set_task(
            task_id,
            message=f"Erreur: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )

        log_to_db('welcome_to_the_jungle', 'ERROR', f"❌ Exception après {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
        
        if job_store.enabled:
            job_store.finish(task_id, STATUS_FAILED, error_text=error_text)
        
        if ENABLE_GRAFANA_LOGS:
            emit_structured_log({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "job_failed",
                "run_id": task_id,
                "source": "wttj",
                "status": STATUS_FAILED,
                "progress_pct": None,
                "records_count": 0,
                "pages_count": 0,
                "errors_count": 1,
                "duration_sec": round(duration_sec, 2)
            })


def run_collect_sitemaps_task(task_id: str, delay: float, max_results: int):
    """Wrapper for collecting WTTJ sitemap URLs with status updates"""
    import time

    start_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)

    if ENABLE_GRAFANA_LOGS:
        emit_structured_log({
            "timestamp": started_at.isoformat(),
            "event_type": "job_started",
            "run_id": task_id,
            "source": "wttj_sitemaps",
            "status": STATUS_RUNNING,
            "progress_pct": 0,
            "records_count": 0,
            "pages_count": 0,
            "errors_count": 0
        })

    try:
        log_to_db('wttj_sitemaps', 'INFO', "📥 Début de la collecte des sitemaps WTTJ", task_id=task_id)
        
        result = collect_sitemap_urls(
            query="",
            entreprise="",
            ville="",
            max_results=max_results,
            delay=delay
        )
        
        duration_sec = time.monotonic() - start_monotonic
        
        if result.get("success"):
            urls_count = result.get("urls_count", 0)
            storage_key = result.get("storage_key")
            
            result_payload = {
                "urls_count": urls_count,
                "storage_key": storage_key,
                "elapsed_s": duration_sec
            }
            
            set_task(
                task_id,
                progress_pct=100,
                message=f"Collecte réussie: {urls_count} URLs collectées",
                records_count=urls_count,
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )

            log_to_db(
                'wttj_sitemaps',
                'INFO',
                f"✅ {urls_count} URLs collectées - {format_eta(duration_sec)} (storage: {storage_key})",
                task_id=task_id,
                duration_sec=round(duration_sec, 2),
                records_count=urls_count,
                storage_key=storage_key
            )
            
            if job_store.enabled:
                job_store.finish(task_id, STATUS_SUCCESS, result=result_payload)
            
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_finished",
                    "run_id": task_id,
                    "source": "wttj_sitemaps",
                    "status": STATUS_SUCCESS,
                    "progress_pct": 100,
                    "records_count": urls_count,
                    "pages_count": 0,
                    "errors_count": 0,
                    "duration_sec": round(duration_sec, 2)
                })
        else:
            error_text = result.get("error", "Erreur inconnue")
            
            set_task(
                task_id,
                message=f"Échec de la collecte: {error_text}",
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )

            log_to_db('wttj_sitemaps', 'ERROR', f"❌ Échec de la collecte: {error_text}", task_id=task_id, error=error_text)
            
            if job_store.enabled:
                job_store.finish(task_id, STATUS_FAILED, result=result, error_text=error_text)
            
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_failed",
                    "run_id": task_id,
                    "source": "wttj_sitemaps",
                    "status": STATUS_FAILED,
                    "progress_pct": None,
                    "records_count": 0,
                    "pages_count": 0,
                    "errors_count": 1,
                    "duration_sec": round(duration_sec, 2)
                })

    except Exception as e:
        duration_sec = time.monotonic() - start_monotonic
        error_text = str(e)
        
        set_task(
            task_id,
            message=f"Erreur: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )

        log_to_db('wttj_sitemaps', 'ERROR', f"❌ Exception après {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
        
        if job_store.enabled:
            job_store.finish(task_id, STATUS_FAILED, error_text=error_text)
        
        if ENABLE_GRAFANA_LOGS:
            emit_structured_log({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "job_failed",
                "run_id": task_id,
                "source": "wttj_sitemaps",
                "status": STATUS_FAILED,
                "progress_pct": None,
                "records_count": 0,
                "pages_count": 0,
                "errors_count": 1,
                "duration_sec": round(duration_sec, 2)
            })


def run_wttj_job_opt_task(
    task_id: str,
    mode: str,
    max_urls: int,
    workers: int,
    part_size: int,
    delay: float,
    force_download_urls: bool
):
    """Wrapper for ingesting WTTJ jobs optimized (API REST) with status updates"""
    import time

    start_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)

    if ENABLE_GRAFANA_LOGS:
        emit_structured_log({
            "timestamp": started_at.isoformat(),
            "event_type": "job_started",
            "run_id": task_id,
            "source": "wttj_opt",
            "status": STATUS_RUNNING,
            "progress_pct": 0,
            "records_count": 0,
            "pages_count": 0,
            "errors_count": 0
        })

    def update_progress(segment: str, current: int, total: int, ok: int, ko: int):
        """Callback to update progress in real-time"""
        progress_pct = int((current / total) * 100) if total > 0 else 0
        set_task(
            task_id,
            progress_pct=progress_pct,
            message=f"Ingestion optimisée: {current}/{total} URLs traitées (✓{ok} ✗{ko})",
            pages_count=current,
            errors_count=ko,
        )
    
    try:
        log_to_db('wttj_opt', 'INFO', "📥 Début de l'ingestion WTTJ optimisée (API REST)", task_id=task_id)
        
        result = ingest_welcome_to_the_jungle_opt(
            storage=None,
            mode=mode,
            max_urls=max_urls,
            workers=workers,
            part_size=part_size,
            delay=delay,
            provided_run_id=task_id,
            progress_callback=update_progress,
            force_download_urls=force_download_urls
        )
        
        duration_sec = time.monotonic() - start_monotonic
        
        if result.get("success"):
            # Extract stats from jobs_opt segment
            jobs_opt = result.get("jobs_opt", {})
            urls_processed = jobs_opt.get("processed", 0)
            urls_ok = jobs_opt.get("ok", 0)
            urls_ko = jobs_opt.get("ko", 0)
            records_written = jobs_opt.get("written", 0)
            
            result_payload = {
                "run_id": result.get("run_id"),
                "dt": result.get("dt"),
                "mode": result.get("mode"),
                "urls_total": urls_processed,
                "urls_processed": urls_processed,
                "urls_ok": urls_ok,
                "urls_ko": urls_ko,
                "records_written": records_written,
                "elapsed_s": result.get("elapsed_s"),
                "storage_prefix": f"dt={result.get('dt')}/run_id={result.get('run_id')}"
            }

            set_task(
                task_id,
                progress_pct=100,
                message=f"Ingestion optimisée réussie: {records_written} records écrits",
                records_count=records_written,
                pages_count=urls_processed,
                errors_count=urls_ko,
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )

            log_to_db(
                'wttj_opt',
                'INFO',
                (
                    f"✅ {records_written} records écrits "
                    f"({urls_processed} URLs, {urls_ko} erreurs) - {format_eta(duration_sec)}"
                ),
                task_id=task_id,
                duration_sec=round(duration_sec, 2),
                records_count=records_written,
                error_count=urls_ko,
                urls_processed=urls_processed
            )
            
            if job_store.enabled:
                job_store.finish(task_id, STATUS_SUCCESS, result=result_payload)
            
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_finished",
                    "run_id": task_id,
                    "source": "wttj_opt",
                    "status": STATUS_SUCCESS,
                    "progress_pct": 100,
                    "records_count": records_written,
                    "pages_count": urls_processed,
                    "errors_count": urls_ko,
                    "duration_sec": round(duration_sec, 2)
                })
        else:
            error_text = result.get("error", "Erreur inconnue")
            set_task(
                task_id,
                message=f"Échec de l'ingestion optimisée: {error_text}",
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )

            log_to_db('wttj_opt', 'ERROR', f"❌ Échec de l'ingestion: {error_text}", task_id=task_id, error=error_text)
            
            if job_store.enabled:
                job_store.finish(task_id, STATUS_FAILED, result=result, error_text=error_text)
            
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_failed",
                    "run_id": task_id,
                    "source": "wttj_opt",
                    "status": STATUS_FAILED,
                    "progress_pct": None,
                    "records_count": 0,
                    "pages_count": 0,
                    "errors_count": 1,
                    "duration_sec": round(duration_sec, 2)
                })

    except Exception as e:
        duration_sec = time.monotonic() - start_monotonic
        error_text = str(e)
        
        set_task(
            task_id,
            message=f"Erreur: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )

        log_to_db('wttj_opt', 'ERROR', f"❌ Exception après {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
        
        if job_store.enabled:
            job_store.finish(task_id, STATUS_FAILED, error_text=error_text)
        
        if ENABLE_GRAFANA_LOGS:
            emit_structured_log({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "job_failed",
                "run_id": task_id,
                "source": "wttj_opt",
                "status": STATUS_FAILED,
                "progress_pct": None,
                "records_count": 0,
                "pages_count": 0,
                "errors_count": 1,
                "duration_sec": round(duration_sec, 2)
            })


def run_merge_datasets_task(task_id: str, ft_prefix: Optional[str], wttj_prefix: Optional[str], 
                             output_prefix: Optional[str], output_format: str):
    """Wrapper pour la fusion des datasets avec mise à jour du statut"""
    
    import time

    start_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)

    if ENABLE_GRAFANA_LOGS:
        emit_structured_log({
            "timestamp": started_at.isoformat(),
            "event_type": "job_started",
            "run_id": task_id,
            "source": "merge",
            "status": STATUS_RUNNING,
            "progress_pct": 0,
            "records_count": 0,
            "pages_count": 0,
            "errors_count": 0
        })
    
    def update_progress(step: str, message: str):
        """Callback pour mettre à jour la progression en temps réel"""
        set_task(
            task_id,
            message=message,
            current_step=step
        )
    
    try:
        log_to_db('merge_datasets', 'INFO', f"🔀 Début de la fusion FT + WTTJ", task_id=task_id)
        result = merge_ft_wttj_datasets(
            ft_prefix=ft_prefix,
            wttj_prefix=wttj_prefix,
            output_prefix=output_prefix,
            output_format=output_format,
            progress_callback=update_progress
        )
        
        duration_sec = time.monotonic() - start_monotonic
        
        if result["success"]:
            result_payload = {
                "output_key": result.get("output_key"),
                "output_format": result.get("output_format"),
                "ft_prefix": result.get("ft_prefix"),
                "wttj_prefix": result.get("wttj_prefix"),
                "total_offers": result.get("total_offers"),
                "ft_offers": result.get("ft_offers"),
                "wttj_offers": result.get("wttj_offers"),
                "offers_with_rome": result.get("offers_with_rome"),
                "unique_rome_codes": result.get("unique_rome_codes"),
                "elapsed_s": result.get("elapsed_s")
            }

            set_task(
                task_id,
                progress_pct=100,
                message=result["message"],
                records_count=result.get("total_offers", 0),
                errors_count=0,
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )

            log_to_db(
                'merge_datasets',
                'INFO',
                f"✅ Fusion terminée: {result.get('total_offers')} offres (FT: {result.get('ft_offers')}, WTTJ: {result.get('wttj_offers')}) - {duration_sec:.2f}s",
                task_id=task_id,
                duration_sec=round(duration_sec, 2),
                records_count=result.get('total_offers', 0),
                ft_offers=result.get('ft_offers', 0),
                wttj_offers=result.get('wttj_offers', 0)
            )
            
            if job_store.enabled:
                job_store.finish(task_id, STATUS_SUCCESS, result=result_payload)
            
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_finished",
                    "run_id": task_id,
                    "source": "merge",
                    "status": STATUS_SUCCESS,
                    "progress_pct": 100,
                    "records_count": result.get("total_offers", 0),
                    "pages_count": 0,
                    "errors_count": 0,
                    "duration_sec": round(duration_sec, 2)
                })
        else:
            error_text = result.get("error")
            
            set_task(
                task_id,
                message=result.get("message", "Échec de la fusion"),
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )
            
            log_to_db('merge_datasets', 'ERROR', f"❌ Échec: {result.get('error')}", task_id=task_id, error=error_text)
            
            if job_store.enabled:
                job_store.finish(task_id, STATUS_FAILED, result=result, error_text=error_text)
            
            if ENABLE_GRAFANA_LOGS:
                emit_structured_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "job_failed",
                    "run_id": task_id,
                    "source": "merge",
                    "status": STATUS_FAILED,
                    "progress_pct": None,
                    "records_count": result.get("total_offers", 0),
                    "pages_count": 0,
                    "errors_count": 1,
                    "duration_sec": round(duration_sec, 2)
                })
    except Exception as e:
        duration_sec = time.monotonic() - start_monotonic
        error_text = str(e)
        
        set_task(
            task_id,
            message=f"Erreur: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        
        log_to_db('merge_datasets', 'ERROR', f"❌ Exception après {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
        
        if job_store.enabled:
            job_store.finish(task_id, STATUS_FAILED, error_text=error_text)
        
        if ENABLE_GRAFANA_LOGS:
            emit_structured_log({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "job_failed",
                "run_id": task_id,
                "source": "merge",
                "status": STATUS_FAILED,
                "progress_pct": None,
                "records_count": 0,
                "pages_count": 0,
                "errors_count": 1,
                "duration_sec": round(duration_sec, 2)
            })


@app.get(
    "/health",
    tags=["Monitoring"],
    summary="Health check de l'API",
    description="Vérifie que l'API est opérationnelle et retourne les informations sur le modèle chargé"
)
def health():
    """Health check de l'API avec informations sur le modèle et le backend de storage"""
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": ARTIFACTS.get("version"),
        "storage_backend": os.getenv("STORAGE_BACKEND", "local"),
    }


@app.get(
    "/jobs",
    tags=["Monitoring"],
    summary="Liste des job runs",
    description="Retourne les runs persistés dans JobStore"
)
def list_jobs(
    source: Optional[str] = Query(None, description="Filtrer par source"),
    status: Optional[str] = Query(None, description="Filtrer par statut"),
    limit: int = Query(50, ge=1, le=200, description="Nombre maximum de résultats")
):
    if not job_store.enabled:
        raise HTTPException(status_code=503, detail="JobStore disabled")
    return {
        "status": "ok",
        "items": job_store.list_runs(source=source, status=status, limit=limit)
    }


@app.get(
    "/jobs/{run_id}",
    tags=["Monitoring"],
    summary="Détail d'un job run",
    description="Retourne un run persistant par run_id"
)
def get_job(run_id: str):
    if not job_store.enabled:
        raise HTTPException(status_code=503, detail="JobStore disabled")
    job = job_store.get_run(run_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {run_id} not found")
    return job

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Classification ROME"],
    summary="Prédire le code ROME d'une offre d'emploi",
    description="""Classifie automatiquement une offre d'emploi selon le référentiel ROME.

Fournissez au minimum un intitulé ou une description de poste.
Le modèle retourne le code ROME le plus probable ainsi qu'un top-K des codes les plus pertinents.
"""
)
def predict(req: PredictRequest):
    """Classification d'une offre d'emploi selon le référentiel ROME.

**Paramètres:**

- **intitule**: Titre du poste (optionnel mais recommandé)
- **description**: Description détaillée (optionnel mais recommandé)
- **competences**: Liste de compétences techniques (optionnel)

**Exemple:**

```json
{
  "intitule": "Data Scientist",
  "description": "Analyse de données et machine learning",
  "competences": ["Python", "Scikit-learn", "SQL"]
}
```
"""
    logger.info(f"Requête de prédiction - Intitulé: {req.intitule[:50] if req.intitule else 'N/A'}")

    if not ARTIFACTS:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts are not loaded yet. "
                "Run training or upload model artifacts, then restart API."
            ),
        )
    
    text = build_text_payload(
        intitule=req.intitule,
        description=req.description,
        competences=req.competences,
    )

    pred = predict_top_k(ARTIFACTS, text, top_k=TOP_K, rome_index=rome_model)
    
    logger.info(f"Prédiction réussie - Code ROME: {pred.get('rome_pred', 'N/A')} / {pred.get('rome_label', 'N/A')} ")
    
    return pred


# =====================================
# Endpoints d'ingestion
# =====================================

@app.post(
    "/ingest/rome-metiers",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingérer les codes ROME métiers",
    description="""Déclenche l'ingestion de la nomenclature complète des codes ROME métiers depuis l'API France Travail.

Les données sont stockées dans `bronze/rome/rome_metiers.jsonl` (environ 532 codes ROME).

**Modes d'exécution:**

- **Synchrone** (background=false): Attend la fin de l'ingestion et retourne le résultat
- **Asynchrone** (background=true): Lance l'ingestion en arrière-plan et retourne immédiatement

**Utilisation:**

```bash
# Synchrone
curl -X POST "http://localhost:8000/ingest/rome-metiers"

# Asynchrone
curl -X POST "http://localhost:8000/ingest/rome-metiers?background=true"
```
"""
)
async def ingest_rome_metiers_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Lancer en arrière-plan")
):
    """Ingestion des codes ROME métiers depuis l'API France Travail.

Récupère tous les codes ROME métiers avec leurs libellés et les stocke
dans le système de storage configuré (local ou S3/MinIO).
"""
    logger.info(f"Requête d'ingestion ROME métiers reçue (background={background})")
    
    if background:
        # Générer un task_id unique
        task_id = utc_run_id()
        
        # Enregistrer la tâche
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_rome_metiers",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": "Ingestion des codes ROME métiers en cours..."
        }
        if job_store.enabled:
            job_store.create(
                run_id=task_id,
                job_type="import",
                source="rome_metiers",
                params={"background": True},
                message="Ingestion des codes ROME métiers en cours...",
            )
        
        # Lancer en arrière-plan avec wrapper
        logger.info("Lancement de l'ingestion en arrière-plan")
        background_tasks.add_task(run_rome_metiers_task, task_id)
        
        return IngestResponse(
            success=True,
            message=f"Ingestion des codes ROME métiers lancée en arrière-plan (task_id: {task_id})",
            key=task_id
        )
    else:
        # Exécution synchrone
        try:
            logger.info("📥 Début de l'ingestion synchrone des codes ROME")
            
            result = ingest_rome_metiers()
            
            if result["success"]:
                logger.info(f"✅ Ingestion réussie: {result['records_count']} codes ROME")
            else:
                logger.error(f"❌ Échec de l'ingestion: {result.get('error')}")
                
            return IngestResponse(**result)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ingestion: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de l'ingestion: {str(e)}"
            )


@app.get(
    "/ingest/status",
    tags=["Ingestion"],
    summary="Statut des opérations d'ingestion",
    description="Liste toutes les opérations d'ingestion disponibles et affiche les tâches en cours"
)
def get_ingest_status():
    """Retourne la liste des opérations d'ingestion disponibles et les tâches actives.

Utile pour découvrir les endpoints d'ingestion et monitorer les tâches en cours d'exécution.
"""
    # Nettoyer les tâches terminées depuis plus de 5 minutes
    current_time = datetime.now(timezone.utc)
    tasks_to_remove = []
    for task_id, task_info in ACTIVE_TASKS.items():
        if task_info.get("status") == STATUS_SUCCESS:
            completed_at = task_info.get("completed_at")
            if completed_at and (current_time - completed_at).total_seconds() > 300:
                tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        del ACTIVE_TASKS[task_id]
    
    # Filtrer les tâches d'ingestion
    ingestion_operations = [
        "ingest_rome_metiers",
        "ingest_france_travail_offers",
        "ingest_welcome_to_jungle",
        "collect_wttj_sitemaps",
        "ingest_wttj_opt"
    ]
    
    return {
        "status": "ok",
        "active_tasks": [
            {
                "task_id": task_id,
                "operation": task_info.get("operation"),
                "status": task_info.get("status"),
                "started_at": task_info.get("started_at").isoformat() if task_info.get("started_at") else None,
                "progress_pct": task_info.get("progress_pct"),
                "message": task_info.get("message")
            }
            for task_id, task_info in ACTIVE_TASKS.items()
            if task_info.get("operation") in ingestion_operations
        ],
        "available_operations": [
            {
                "endpoint": "POST /ingest/rome-metiers",
                "description": "Ingestion des codes ROME métiers depuis France Travail",
                "params": ["background (bool, optionnel)"]
            },
            {
                "endpoint": "POST /ingest/france-travail-offers",
                "description": "Ingestion des offres d'emploi France Travail (couche bronze)",
                "params": [
                    "background (bool, optionnel)",
                    "max_rome_codes (int, optionnel)",
                    "window_days (int, optionnel)"
                ]
            },
            {
                "endpoint": "POST /ingest/welcome-to-jungle",
                "description": "Ingestion des offres d'emploi Welcome to the Jungle (couche bronze)",
                "params": [
                    "background (bool, optionnel)",
                    "mode (str, optionnel: new/resume/incremental)",
                    "max_jobs (int, optionnel)",
                    "max_companies (int, optionnel)"
                ]
            },
            {
                "endpoint": "POST /ingest/welcome-to-the-jungle/sitemaps",
                "description": "Collecte des URLs WTTJ depuis les sitemaps XML",
                "params": [
                    "background (bool, optionnel)",
                    "delay (float, optionnel)",
                    "max_results (int, optionnel)"
                ]
            },
            {
                "endpoint": "POST /ingest/welcome-to-the-jungle/jobs-optimized",
                "description": "Ingestion optimisée WTTJ via API REST avec fallback JSON-LD",
                "params": [
                    "background (bool, optionnel)",
                    "mode (str, optionnel: new/resume/incremental)",
                    "max_urls (int, optionnel)",
                    "workers (int, optionnel)",
                    "part_size (int, optionnel)",
                    "delay (float, optionnel)",
                    "force_download_urls (bool, optionnel)"
                ]
            }
        ]
    }


@app.get(
    "/data/status",
    tags=["Data Processing"],
    summary="Statut des opérations de traitement de données",
    description="Liste toutes les opérations de data processing disponibles et affiche les tâches en cours"
)
def get_data_status():
    """Retourne la liste des opérations de data processing disponibles et les tâches actives.

Utile pour découvrir les endpoints de traitement et monitorer les tâches en cours d'exécution.
"""
    # Nettoyer les tâches terminées depuis plus de 5 minutes
    current_time = datetime.now(timezone.utc)
    tasks_to_remove = []
    for task_id, task_info in ACTIVE_TASKS.items():
        if task_info.get("status") == STATUS_SUCCESS:
            completed_at = task_info.get("completed_at")
            if completed_at and (current_time - completed_at).total_seconds() > 300:
                tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        del ACTIVE_TASKS[task_id]
    
    # Filtrer les tâches de data processing
    data_operations = [
        "merge_datasets",
        "normalize_wttj_jobs",
        "normalize_ft_jobs",
    ]
    
    return {
        "status": "ok",
        "active_tasks": [
            {
                "task_id": task_id,
                "operation": task_info.get("operation"),
                "status": task_info.get("status"),
                "started_at": task_info.get("started_at").isoformat() if task_info.get("started_at") else None,
                "progress_pct": task_info.get("progress_pct"),
                "message": task_info.get("message")
            }
            for task_id, task_info in ACTIVE_TASKS.items()
            if task_info.get("operation") in data_operations
        ],
        "available_operations": [
            {
                "endpoint": "POST /data/normalize-ft-jobs",
                "description": "Normalisation FT bronze vers silver (schema canonique)",
                "params": [
                    "background (bool, optionnel)",
                    "dt (str, optionnel: YYYY-MM-DD/latest)",
                    "output_format (str, optionnel: parquet/jsonl/csv)"
                ]
            },
            {
                "endpoint": "POST /data/normalize-wttj-jobs",
                "description": "Normalisation WTTJ bronze vers silver (avec enrichissement ROME)",
                "params": [
                    "background (bool, optionnel)",
                    "dt (str, optionnel: YYYY-MM-DD/latest)",
                    "output_format (str, optionnel: parquet/jsonl/csv)"
                ]
            },
            {
                "endpoint": "POST /data/merge-datasets",
                "description": "Fusion des datasets FT et WTTJ avec ROME codes",
                "params": [
                    "background (bool, optionnel)",
                    "ft_prefix (str, optionnel)",
                    "wttj_prefix (str, optionnel)",
                    "output_prefix (str, optionnel)",
                    "output_format (str, optionnel: parquet/jsonl/csv)"
                ]
            }
        ]
    }


@app.get(
    "/tasks/{task_id}",
    tags=["Monitoring"],
    summary="Détails d'une tâche",
    description="Récupère les informations détaillées d'une tâche (ingestion, merge, etc.)"
)
def get_task_details_generic(task_id: str):
    """Retourne les détails complets d'une tâche asynchrone.

**Utilisation:**

```bash
# Récupérer le statut d'une tâche d'ingestion
curl http://localhost:8000/tasks/ft-offers-20260223T214500

# Récupérer le statut d'une tâche de merge
curl http://localhost:8000/tasks/merge-20260225T143000
```

**Statuts possibles:**

- `running`: Tâche en cours d'exécution
- `completed`: Tâche terminée avec succès
- `failed`: Tâche échouée
"""
    if task_id not in ACTIVE_TASKS:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task_info = ACTIVE_TASKS[task_id]
    return {
        "task_id": task_id,
        "operation": task_info.get("operation"),
        "status": task_info.get("status"),
        "started_at": task_info.get("started_at").isoformat() if task_info.get("started_at") else None,
        "completed_at": task_info.get("completed_at").isoformat() if task_info.get("completed_at") else None,
        "progress_pct": task_info.get("progress_pct"),
        "message": task_info.get("message"),
        "params": task_info.get("params"),
        "result": task_info.get("result"),
        "error": task_info.get("error")
    }


@app.get(
    "/ingest/tasks/{task_id}",
    tags=["Ingestion"],
    summary="Détails d'une tâche d'ingestion (déprécié)",
    description="⚠️ Déprécié: Utiliser /tasks/{task_id} à la place. Récupère les informations détaillées d'une tâche d'ingestion spécifique",
    deprecated=True
)
def get_task_details(task_id: str):
    """Retourne les détails complets d'une tâche d'ingestion.

⚠️ **Cet endpoint est déprécié**, utilisez `/tasks/{task_id}` à la place.

**Utilisation:**

```bash
# Après avoir lancé une ingestion en arrière-plan, récupérer son statut
curl http://localhost:8000/ingest/tasks/ft-offers-20260223T214500
```
"""
    return get_task_details_generic(task_id)


@app.post(
    "/ingest/france-travail-offers",
    response_model=IngestOffersResponse,
    tags=["Ingestion"],
    summary="Ingérer les offres d'emploi France Travail",
    description="""Déclenche l'ingestion complète des offres d'emploi France Travail en couche bronze.

Cette opération peut être longue (plusieurs heures pour tous les codes ROME).
Il est recommandé de l'exécuter en mode asynchrone.

**Modes d'exécution:**

- **Synchrone** (background=false): Attend la fin complète de l'ingestion
- **Asynchrone** (background=true): Lance l'ingestion en arrière-plan

**Paramètres de contrôle:**

- **max_rome_codes**: Limite le nombre de codes ROME à traiter (0 = tous, utile pour tests)
- **window_days**: Taille des fenêtres temporelles en jours (défaut: 7)
- **max_windows**: Nombre maximum de fenêtres temporelles (défaut: 260)

**Utilisation:**

```bash
# Ingestion complète en arrière-plan
curl -X POST "http://localhost:8000/ingest/france-travail-offers?background=true"

# Test avec seulement 5 codes ROME
curl -X POST "http://localhost:8000/ingest/france-travail-offers?max_rome_codes=5"
```
"""
)
async def ingest_france_travail_offers_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Lancer en arrière-plan"),
    max_rome_codes: int = Query(0, description="Limiter le nombre de codes ROME (0 = tous)"),
    window_days: int = Query(7, description="Taille des fenêtres temporelles en jours"),
    max_windows: int = Query(260, description="Nombre maximum de fenêtres temporelles"),
    binary_split_min_seconds: int = Query(3600, description="Taille minimale de fenêtre pour split binaire")
):
    """Ingestion des offres d'emploi France Travail en couche bronze.

Récupère toutes les offres d'emploi pour chaque code ROME et les stocke
dans le système de storage configuré (local ou S3/MinIO).
"""
    logger.info(f"Requête d'ingestion offres FT reçue (background={background}, max_rome_codes={max_rome_codes})")
    
    if background:
        # Générer un task_id unique
        task_id = utc_run_id()
        
        # Enregistrer la tâche
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_france_travail_offers",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Ingestion des offres France Travail en cours (max_rome_codes: {max_rome_codes or 'tous'})...",
            "params": {
                "max_rome_codes": max_rome_codes,
                "window_days": window_days,
                "max_windows": max_windows
            }
        }
        if job_store.enabled:
            job_store.create(
                run_id=task_id,
                job_type="import",
                source="france_travail_offers",
                params={
                    "background": True,
                    "max_rome_codes": max_rome_codes,
                    "window_days": window_days,
                    "max_windows": max_windows,
                    "binary_split_min_seconds": binary_split_min_seconds,
                },
                message=f"Ingestion des offres France Travail en cours (max_rome_codes: {max_rome_codes or 'tous'})...",
            )
        
        # Lancer en arrière-plan avec wrapper
        logger.info("Lancement de l'ingestion des offres en arrière-plan")
        background_tasks.add_task(
            run_france_travail_offers_task,
            task_id,
            window_days,
            max_windows,
            binary_split_min_seconds,
            max_rome_codes
        )
        
        return IngestOffersResponse(
            success=True,
            message=f"Ingestion des offres France Travail lancée en arrière-plan (task_id: {task_id})",
            run_id=task_id
        )
    else:
        # Exécution synchrone
        try:
            logger.info("📥 Début de l'ingestion synchrone des offres France Travail")
            result = ingest_france_travail_offers(
                storage=None,
                client=None,
                window_days=window_days,
                max_windows=max_windows,
                binary_split_min_seconds=binary_split_min_seconds,
                max_rome_codes=max_rome_codes,
                logger_override=None
            )
            
            if result["success"]:
                logger.info(f"✅ Ingestion réussie: {result['written']} offres, {result['rome_processed']} codes ROME")
            else:
                logger.error(f"❌ Échec de l'ingestion: {result.get('error')}")
                
            return IngestOffersResponse(**result)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ingestion des offres: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de l'ingestion: {str(e)}"
            )


@app.post(
    "/ingest/welcome-to-jungle",
    response_model=IngestWTTJResponse,
    tags=["Ingestion"],
    summary="Ingérer les offres Welcome to the Jungle",
    description="""Déclenche l'ingestion des offres d'emploi Welcome to the Jungle en couche bronze.

Cette opération collecte les URLs depuis les sitemaps et extrait les données
des pages jobs et companies.

**Modes d'exécution:**

- **Synchrone** (background=false): Attend la fin complète de l'ingestion
- **Asynchrone** (background=true): Lance l'ingestion en arrière-plan

**Modes d'ingestion:**

- **new**: Nouveau run avec run_id généré, pas de skip
- **resume**: Reprend un run existant (nécessite run_id)  
- **incremental**: Skip les URLs déjà traitées dans un run précédent

**Paramètres de contrôle:**

- **max_jobs**: Limite le nombre de jobs à traiter (0 = tous)
- **max_companies**: Limite le nombre de companies à traiter (0 = tous)

**Utilisation:**

```bash
# Ingestion complète en arrière-plan
curl -X POST "http://localhost:8000/ingest/welcome-to-jungle?background=true"

# Test avec 100 jobs et 50 companies
curl -X POST "http://localhost:8000/ingest/welcome-to-jungle?background=true&max_jobs=100&max_companies=50"
```
"""
)
async def ingest_wttj_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Lancer en arrière-plan"),
    mode: str = Query("new", description="Mode d'ingestion (new, resume, incremental)"),
    max_jobs: int = Query(0, description="Limiter le nombre de jobs (0 = tous)"),
    max_companies: int = Query(0, description="Limiter le nombre de companies (0 = tous)"),
    workers: int = Query(10, description="Nombre de workers concurrents (défaut: 10)"),
    part_size: int = Query(500, description="Taille des chunks JSONL en records (défaut: 500)"),
    provided_run_id: str | None = Query(None, description="Run ID à utiliser en mode resume"),
    resume_from_run_id: str | None = Query(None, description="Run ID source pour mode incremental"),
):
    """Ingestion des offres d'emploi Welcome to the Jungle en couche bronze.

Collecte les URLs depuis les sitemaps et extrait les données structurées
des pages jobs et companies pour stockage en bronze.
"""
    logger.info(
        "Requête d'ingestion WTTJ reçue (background=%s, mode=%s, max_jobs=%s, max_companies=%s, workers=%s, part_size=%s, provided_run_id=%s, resume_from_run_id=%s)",
        background,
        mode,
        max_jobs,
        max_companies,
        workers,
        part_size,
        provided_run_id,
        resume_from_run_id,
    )
    
    if background:
        # Générer un task_id unique
        task_id = utc_run_id()
        
        # Enregistrer la tâche
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_welcome_to_jungle",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Ingestion WTTJ en cours (mode: {mode}, jobs: {max_jobs or 'tous'}, companies: {max_companies or 'tous'}, workers: {workers}, part_size: {part_size})...",
            "params": {
                "mode": mode,
                "max_jobs": max_jobs,
                "max_companies": max_companies,
                "workers": workers,
                "part_size": part_size,
                "provided_run_id": provided_run_id,
                "resume_from_run_id": resume_from_run_id,
            }
        }
        if job_store.enabled:
            job_store.create(
                run_id=task_id,
                job_type="import",
                source="wttj",
                params={
                    "background": True,
                    "mode": mode,
                    "max_jobs": max_jobs,
                    "max_companies": max_companies,
                    "workers": workers,
                    "part_size": part_size,
                    "provided_run_id": provided_run_id,
                    "resume_from_run_id": resume_from_run_id,
                },
                message=f"Ingestion WTTJ en cours (mode: {mode}, jobs: {max_jobs or 'tous'}, companies: {max_companies or 'tous'}, workers: {workers}, part_size: {part_size})...",
            )
        
        # Lancer en arrière-plan avec wrapper
        logger.info("Lancement de l'ingestion WTTJ en arrière-plan")
        background_tasks.add_task(
            run_welcome_to_jungle_task,
            task_id,
            mode,
            max_jobs,
            max_companies,
            workers,
            part_size,
            provided_run_id or task_id,
            resume_from_run_id,
        )
        
        return IngestWTTJResponse(
            success=True,
            message=f"Ingestion WTTJ lancée en arrière-plan (task_id: {task_id})",
            run_id=task_id
        )
    else:
        # Exécution synchrone
        try:
            logger.info("📥 Début de l'ingestion synchrone WTTJ")
            result = ingest_welcome_to_the_jungle(
                storage=None,
                mode=mode,
                max_jobs=max_jobs,
                max_companies=max_companies,
                workers=workers,
                part_size=part_size,
                provided_run_id=provided_run_id,
                resume_from_run_id=resume_from_run_id,
            )
            
            if result["success"]:
                logger.info(f"✅ Ingestion réussie: {result.get('total_written')} records")
            else:
                logger.error(f"❌ Échec de l'ingestion: {result.get('error')}")
                
            return IngestWTTJResponse(**result)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ingestion WTTJ: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de l'ingestion: {str(e)}"
            )


@app.post(
    "/ingest/welcome-to-the-jungle/sitemaps",
    response_model=CollectSitemapsResponse,
    tags=["Ingestion"],
    summary="Collecter les URLs depuis les sitemaps WTTJ",
    description="""Collecte les URLs depuis les sitemaps Welcome to the Jungle.

Cette opération récupère les URLs des pages jobs depuis les sitemaps XML de WTTJ
et les stocke dans le système de stockage bronze pour traitement ultérieur.

**Modes d'exécution:**

- **Synchrone** (background=false): Attend la fin complète de la collecte
- **Asynchrone** (background=true): Lance la collecte en arrière-plan

**Paramètres de contrôle:**

- **delay**: Délai en secondes entre chaque requête (défaut: 0.5s)
- **max_results**: Nombre maximum d'URLs à collecter (0 = toutes)

**Utilisation:**

```bash
# Collecte complète avec délai par défaut (synchrone)
curl -X POST "http://localhost:8000/ingest/welcome-to-the-jungle/sitemaps"

# Collecte en arrière-plan
curl -X POST "http://localhost:8000/ingest/welcome-to-the-jungle/sitemaps?background=true"

# Collecte limitée à 1000 URLs avec délai de 1s
curl -X POST "http://localhost:8000/ingest/welcome-to-the-jungle/sitemaps?max_results=1000&delay=1.0"
```
"""
)
async def collect_sitemaps_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Lancer en arrière-plan"),
    delay: float = Query(0.5, description="Délai entre chaque requête (secondes)"),
    max_results: int = Query(0, description="Nombre max d'URLs à collecter (0 = toutes)")
):
    """Collecte les URLs depuis les sitemaps WTTJ et les stocke en bronze.

Cette opération est généralement rapide (quelques secondes pour ~10k URLs).
Les URLs sont stockées dans bronze/welcometothejungle/sitemap/urls.txt.
"""
    logger.info(f"Requête de collecte sitemaps reçue (background={background}, delay={delay}, max_results={max_results})")
    
    if background:
        # Générer un task_id unique
        task_id = utc_run_id()
        
        # Enregistrer la tâche
        ACTIVE_TASKS[task_id] = {
            "operation": "collect_wttj_sitemaps",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Collecte des sitemaps WTTJ en cours (max_results: {max_results or 'toutes'}, delay: {delay}s)...",
            "params": {
                "delay": delay,
                "max_results": max_results
            }
        }
        if job_store.enabled:
            job_store.create(
                run_id=task_id,
                job_type="import",
                source="wttj_sitemaps",
                params={
                    "background": True,
                    "delay": delay,
                    "max_results": max_results,
                },
                message=f"Collecte des sitemaps WTTJ en cours (max_results: {max_results or 'toutes'}, delay: {delay}s)...",
            )
        
        # Lancer en arrière-plan avec wrapper
        logger.info("Lancement de la collecte sitemaps WTTJ en arrière-plan")
        background_tasks.add_task(
            run_collect_sitemaps_task,
            task_id,
            delay,
            max_results
        )
        
        return CollectSitemapsResponse(
            success=True,
            message=f"Collecte sitemaps WTTJ lancée en arrière-plan (task_id: {task_id})",
            urls_count=0,
            storage_key=task_id,
            elapsed_s=None,
            error=None
        )
    else:
        # Exécution synchrone
        import time
        
        try:
            start_time = time.time()
            
            # Appel de la fonction de collecte
            logger.info("📥 Début de la collecte synchrone des sitemaps WTTJ")
            result = collect_sitemap_urls(
                query="",
                entreprise="",
                ville="",
                max_results=max_results,
                delay=delay
            )
            
            elapsed_s = time.time() - start_time
            
            if result.get("success"):
                urls_count = result.get("total_processed", 0)
                storage_key = result.get("storage_key")
                
                logger.info(
                    f"✅ Collecte réussie: {urls_count} URLs en {elapsed_s:.2f}s (storage: {storage_key})"
                )
                
                return CollectSitemapsResponse(
                    success=True,
                    message=f"Collecte réussie: {urls_count} URLs collectées",
                    urls_count=urls_count,
                    storage_key=storage_key,
                    elapsed_s=elapsed_s,
                    error=None
                )
            else:
                error_msg = result.get("error", "Erreur inconnue")
                logger.error(f"❌ Échec de la collecte: {error_msg}")
                
                return CollectSitemapsResponse(
                    success=False,
                    message=f"Échec de la collecte: {error_msg}",
                    urls_count=0,
                    storage_key=None,
                    elapsed_s=elapsed_s,
                    error=error_msg
                )
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la collecte sitemaps: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de la collecte des sitemaps: {str(e)}"
            )


@app.post(
    "/ingest/welcome-to-the-jungle/jobs-optimized",
    response_model=IngestWTTJOptResponse,
    tags=["Ingestion"],
    summary="Ingérer les jobs WTTJ via crawl optimisée",
    description="""Déclenche l'ingestion optimisée des offres Welcome to the Jungle via crawl.

Cette version utilise un crawler pour récupérer les données complètes.

**Modes d'exécution:**

- **Synchrone** (background=false): Attend la fin complète de l'ingestion
- **Asynchrone** (background=true): Lance l'ingestion en arrière-plan

**Modes d'ingestion:**

- **new**: Nouveau run avec run_id généré
- **resume**: Reprend un run existant
- **incremental**: Skip les URLs déjà traitées

**Paramètres de contrôle:**

- **max_urls**: Limite le nombre d'URLs à traiter (0 = toutes)
- **workers**: Nombre de workers concurrents (défaut: 8)
- **part_size**: Taille des chunks JSONL (défaut: 500)
- **delay**: Délai entre requêtes par thread en secondes (défaut: 1.0)
- **force_download_urls**: Force le re-téléchargement des URLs depuis les sitemaps

**Utilisation:**

```bash
# Ingestion complète en arrière-plan
curl -X POST "http://localhost:8000/ingest/welcome-to-the-jungle/jobs-optimized?background=true"

# Test avec 1000 URLs et 4 workers
curl -X POST "http://localhost:8000/ingest/welcome-to-the-jungle/jobs-optimized?max_urls=1000&workers=4"

# Mode incremental avec délai personnalisé
curl -X POST "http://localhost:8000/ingest/welcome-to-the-jungle/jobs-optimized?mode=incremental&delay=2.0"
```
"""
)
async def ingest_wttj_jobs_optimized_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Lancer en arrière-plan"),
    mode: str = Query("new", description="Mode d'ingestion (new, resume, incremental)"),
    max_urls: int = Query(0, description="Limiter le nombre d'URLs (0 = toutes)"),
    workers: int = Query(8, description="Nombre de workers concurrents"),
    part_size: int = Query(5000, description="Taille des chunks JSONL en records"),
    delay: float = Query(2.0, description="Délai entre requêtes par thread (secondes)"),
    force_download_urls: bool = Query(True, description="Forcer le re-téléchargement des URLs")
):
    """Ingestion optimisée du crawl de WTTJ.

Utilise un crawler pour récupérer les données complètes.
"""
    logger.info(
        f"Requête d'ingestion WTTJ optimisée reçue (background={background}, mode={mode}, "
        f"max_urls={max_urls}, workers={workers}, delay={delay}, force_download_urls={force_download_urls})"
    )
    
    if background:
        # Générer un task_id unique
        task_id = utc_run_id()
        
        # Enregistrer la tâche
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_wttj_opt",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Ingestion WTTJ optimisée en cours (mode: {mode}, max_urls: {max_urls or 'toutes'}, workers: {workers})...",
            "params": {
                "mode": mode,
                "max_urls": max_urls,
                "workers": workers,
                "part_size": part_size,
                "delay": delay,
                "force_download_urls": force_download_urls
            }
        }
        if job_store.enabled:
            job_store.create(
                run_id=task_id,
                job_type="import",
                source="wttj_opt",
                params={
                    "background": True,
                    "mode": mode,
                    "max_urls": max_urls,
                    "workers": workers,
                    "part_size": part_size,
                    "delay": delay,
                    "force_download_urls": force_download_urls
                },
                message=f"Ingestion WTTJ optimisée en cours (mode: {mode}, max_urls: {max_urls or 'toutes'}, workers: {workers})...",
            )
        
        # Lancer en arrière-plan avec wrapper
        logger.info("Lancement de l'ingestion WTTJ optimisée en arrière-plan")
        background_tasks.add_task(
            run_wttj_job_opt_task,
            task_id,
            mode,
            max_urls,
            workers,
            part_size,
            delay,
            force_download_urls
        )
        
        return IngestWTTJOptResponse(
            success=True,
            message=f"Ingestion WTTJ optimisée lancée en arrière-plan (task_id: {task_id})",
            run_id=task_id,
            mode=mode,
            urls_total=None,
            urls_processed=None,
            urls_ok=None,
            urls_ko=None,
            records_written=None,
            elapsed_s=None,
            error=None
        )
    else:
        # Exécution synchrone
        try:
            logger.info("📥 Début de l'ingestion synchrone WTTJ optimisée")
            result = ingest_welcome_to_the_jungle_opt(
                storage=None,
                mode=mode,
                max_urls=max_urls,
                workers=workers,
                part_size=part_size,
                delay=delay,
                provided_run_id=None,
                progress_callback=None,
                force_download_urls=force_download_urls
            )
            
            if result.get("success"):
                # Extract stats from jobs_opt segment
                jobs_opt = result.get("jobs_opt", {})
                urls_processed = jobs_opt.get("processed", 0)
                urls_ok = jobs_opt.get("ok", 0)
                urls_ko = jobs_opt.get("ko", 0)
                records_written = jobs_opt.get("written", 0)
                
                logger.info(
                    f"✅ Ingestion réussie: {records_written} records écrits "
                    f"({urls_processed} URLs, {urls_ko} erreurs)"
                )
                
                return IngestWTTJOptResponse(
                    success=True,
                    message=result.get("message", ""),
                    run_id=result.get("run_id"),
                    dt=result.get("dt"),
                    mode=result.get("mode"),
                    urls_total=urls_processed,  # Total = processed in this context
                    urls_processed=urls_processed,
                    urls_ok=urls_ok,
                    urls_ko=urls_ko,
                    records_written=records_written,
                    elapsed_s=result.get("elapsed_s"),
                    storage_prefix=f"dt={result.get('dt')}/run_id={result.get('run_id')}",
                    error=None
                )
            else:
                error_msg = result.get("error", "Erreur inconnue")
                logger.error(f"❌ Échec de l'ingestion: {error_msg}")
                
                return IngestWTTJOptResponse(
                    success=False,
                    message=result.get("message", "Échec de l'ingestion"),
                    run_id=result.get("run_id"),
                    dt=result.get("dt"),
                    mode=result.get("mode"),
                    urls_total=None,
                    urls_processed=None,
                    urls_ok=None,
                    urls_ko=None,
                    records_written=None,
                    elapsed_s=result.get("elapsed_s"),
                    storage_prefix=None,
                    error=error_msg
                )
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ingestion WTTJ optimisée: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de l'ingestion: {str(e)}"
            )


@app.post(
    "/data/merge-datasets",
    response_model=MergeDatasetResponse,
    tags=["Data Processing"],
    summary="Fusionner les datasets FT et WTTJ",
    description="""Déclenche la fusion des datasets France Travail et Welcome to the Jungle.

Cette operation lit les donnees deja normalisees en Silver (FT et WTTJ),
les fusionne et les deduplique pour creer un dataset d'entrainement unifie.

**Modes d'exécution:**

- **Synchrone** (background=false): Attend la fin complète de la fusion
- **Asynchrone** (background=true): Lance la fusion en arrière-plan

**Détection automatique:**

Si les préfixes ne sont pas spécifiés, l'API détectera automatiquement les données les plus récentes.

**Formats de sortie:**

- **parquet**: Format binaire optimisé (recommandé)
- **jsonl**: Format JSON Lines texte
- **csv**: Format CSV traditionnel

**Utilisation:**

```bash
# Fusion complète en arrière-plan avec détection auto
curl -X POST "http://localhost:8000/data/merge-datasets?background=true"

# Fusion avec prefixes specifiques
curl -X POST "http://localhost:8000/data/merge-datasets?ft_prefix=dt=2026-02-28/segment=jobs&wttj_prefix=dt=2026-02-28/segment=jobs"

# Fusion avec format de sortie CSV
curl -X POST "http://localhost:8000/data/merge-datasets?output_format=csv"
```
"""
)
async def merge_datasets_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Lancer en arrière-plan"),
    ft_prefix: Optional[str] = Query(None, description="Préfixe des données FT (détection auto si non spécifié)"),
    wttj_prefix: Optional[str] = Query(None, description="Préfixe des données WTTJ (détection auto si non spécifié)"),
    output_prefix: Optional[str] = Query(None, description="Préfixe de sortie (défaut: datasets/ft_wttj_merged)"),
    output_format: str = Query("parquet", description="Format de sortie (parquet, jsonl, csv)")
):
    """Fusion des datasets France Travail et Welcome to the Jungle.

Lit les donnees des couches Silver (FT et WTTJ), harmonise le schema,
fusionne et deduplique pour creer un dataset unifie.

**Étapes:**

1. Détection automatique des préfixes si non spécifiés
2. Lecture des donnees FT Silver
3. Lecture des donnees WTTJ Silver
4. Harmonisation legere du schema canonique
5. Fusion et deduplication par URL
6. Calcul des statistiques
7. Sauvegarde du dataset fusionne
"""
    logger.info(f"Requête de fusion datasets reçue (background={background}, format={output_format})")
    
    if background:
        # Générer un task_id unique
        task_id = utc_run_id()
        
        # Enregistrer la tâche
        ACTIVE_TASKS[task_id] = {
            "operation": "merge_datasets",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Fusion des datasets en cours (format: {output_format})...",
            "params": {
                "ft_prefix": ft_prefix,
                "wttj_prefix": wttj_prefix,
                "output_prefix": output_prefix,
                "output_format": output_format
            }
        }
        if job_store.enabled:
            job_store.create(
                run_id=task_id,
                job_type="data",
                source="merge",
                params={
                    "background": True,
                    "ft_prefix": ft_prefix,
                    "wttj_prefix": wttj_prefix,
                    "output_prefix": output_prefix,
                    "output_format": output_format,
                },
                message=f"Fusion des datasets en cours (format: {output_format})...",
            )
        
        # Lancer en arrière-plan avec wrapper
        logger.info("Lancement de la fusion en arrière-plan")
        background_tasks.add_task(
            run_merge_datasets_task,
            task_id,
            ft_prefix,
            wttj_prefix,
            output_prefix,
            output_format
        )
        
        return MergeDatasetResponse(
            success=True,
            message=f"Fusion des datasets lancée en arrière-plan (task_id: {task_id})",
            output_key=task_id
        )
    else:
        # Exécution synchrone
        try:
            logger.info("🔀 Début de la fusion synchrone des datasets")
            result = merge_ft_wttj_datasets(
                ft_prefix=ft_prefix,
                wttj_prefix=wttj_prefix,
                output_prefix=output_prefix,
                output_format=output_format
            )
            
            if result["success"]:
                logger.info(f"✅ Fusion réussie: {result.get('total_offers')} offres fusionnées")
            else:
                logger.error(f"❌ Échec de la fusion: {result.get('error')}")
                
            return MergeDatasetResponse(**result)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fusion: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de la fusion: {str(e)}"
            )
