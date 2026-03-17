"""
Main API module for ROME classification and data ingestion.

This module defines the FastAPI application, endpoints for prediction
and ingestion, and integrates with the JobStore for tracking long-running tasks.

It also uses log_to_db utility to log ingestion events directly to PostgreSQL,
with graceful degradation if psycopg is not installed.

Structure:
    Section 1 — Configuration & Initialization
    Section 2 — Startup
    Section 3 — Business logic functions
    Section 4 — Background wrappers (tracking + business logic)
    Section 4 — Monitoring endpoints
    Section 5 — Business endpoints
"""

# ============================================================
# SECTION 1 — Configuration & Initialization
# ============================================================

import os
from src.config.env import require_env, get_project_root, load_project_env
load_project_env()  # Safe to call multiple times (idempotent)

# Feature flags and environment configuration
ENABLE_GRAFANA_LOGS = os.getenv("ENABLE_GRAFANA_LOGS", "false").lower() == "true"
JOBSTORE_DSN = os.getenv("JOBSTORE_DSN")
STALE_JOB_MINUTES = int(os.getenv("STALE_JOB_MINUTES", "15"))

# Task status constants
STATUS_RUNNING = "RUNNING"
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"

# Model configuration
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

# ML model
from src.models.predict_model import build_text_payload, load_artifacts, predict_top_k, get_rome_model

# Ingestion modules
from src.ingest.silver.normalize_wttj_jobs import normalize_wttj_jobs
from src.ingest.silver.normalize_ft_jobs import normalize_ft_jobs
from src.ingest.bronze.ingest_france_travail_rome_metiers import ingest_rome_metiers
from src.ingest.bronze.ingest_france_travail_jobs import ingest_france_travail_offers
from src.ingest.bronze.ingest_wttj_jobs import ingest_welcome_to_the_jungle
from src.ingest.bronze.ingest_wttj_collect_urls import collect_sitemap_urls
from src.ingest.bronze.ingest_wttj_job_opt import ingest_welcome_to_the_jungle_opt
from src.ingest.silver.merge_ft_wttj_datasets import merge_ft_wttj_datasets

# Observability & utilities
from src.observability.job_store import JobStore
from src.utils.log_to_db import log_to_db
from src.utils.time_helpers import format_eta, utc_run_id

# API models
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

# ------------------------------------
# FastAPI application
# ------------------------------------
app = FastAPI(
    title="ETL API endpoints for job offer ingestion, cleaning, merging, and enrichment. In order to populate a Medaillon data architecture",
    version="1.0.0",
    description="""
ETL API endpoints for job offer ingestion, cleaning, merging, and enrichment. In order to populate a Medaillon data architecture

## Features

- **ROME Prediction**: Classifies a job offer and returns the corresponding ROME codes
- **ROME Code Ingestion**: Imports the complete ROME job nomenclature
- **Job Offer Ingestion**: Imports job offers from France Travail
- **Job normalization**: Normalizes raw job data from France Travail and Welcome to the Jungle into a unified silver layer
- **Job Merge**: Merges France Travail and Welcome to the Jungle datasets into a unified silver layer
- **Job compute status history**: Compute job offer status history based on publication and presence in the last analytics dataset and new merged datasets
- **Job analytics**: Compute a dataset with current offer status
- **Monitoring**: Status and health checks

## Usage

Use the endpoints below to test the API directly from this interface.
    """,
    contact={
        "name": "Job Market API Support"
    },
    openapi_tags=[
        {"name": "Extraction",        "description": "Ingestion of raw job data from France Travail and Welcome to the Jungle into the bronze layer."},
        {"name": "Transformation",  "description": "Normalization and merging of raw job data into the silver layer."},
        {"name": "Load",  "description": "Loading of processed job data into the final destination."},
        {"name": "Machine Learning", "description": "ROME code classification using the LinearSVC + TF-IDF model."},
        {"name": "Monitoring",       "description": "Health checks, task status, and job history."},
    ]
)

# ------------------------------------
# Global in-memory state
# ------------------------------------

# ML model artifacts — loaded once at startup to avoid reloading MinIO/joblib on each request
ARTIFACTS: Dict[str, Any] = {}
rome_model = None

# In-memory task tracker — holds status of all running/recent background tasks
# Note: lost on container restart; use JobStore for persistence across restarts
ACTIVE_TASKS: Dict[str, Dict[str, Any]] = {}

# ------------------------------------
# Logging setup
# ------------------------------------

def setup_logging():
    """
    Configure logging with console and rotating file handlers,
    plus optional JSON structured logging for Grafana.

    Handlers:
        1. Console — stdout, all levels
        2. logs/api/main.log — rotating file, all levels
        3. logs/api/errors.log — rotating file, ERROR only
        4. logs/api/structured.jsonl — JSON-only, optional (ENABLE_GRAFANA_LOGS)

    Environment variables:
        LOG_LEVEL        — log level (default: INFO)
        LOG_MAX_BYTES    — max file size before rotation (default: 10MB)
        LOG_BACKUP_COUNT — number of rotated files to keep (default: 5)
        ENABLE_GRAFANA_LOGS — enable structured JSON log file (default: false)
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10MB default
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    # Create log directories
    Path("logs/api").mkdir(parents=True, exist_ok=True)
    Path("logs/ingestion").mkdir(parents=True, exist_ok=True)
    Path("logs/prediction").mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Human-readable formatter
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Register base handlers only once (avoid duplicates on hot reload)
    if not root_logger.handlers:

        # 1. Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(standard_formatter)
        console_handler.setLevel(getattr(logging, log_level))

        # 2. Main rotating file
        file_handler = RotatingFileHandler(
            'logs/api/main.log',
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(standard_formatter)
        file_handler.setLevel(getattr(logging, log_level))

        # 3. Error-only rotating file
        error_handler = RotatingFileHandler(
            'logs/api/errors.log',
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding='utf-8'
        )
        error_handler.setFormatter(standard_formatter)
        error_handler.setLevel(logging.ERROR)

        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)

        # Redirect uvicorn loggers to the same handlers for unified log files
        for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uvicorn_logger = logging.getLogger(logger_name)
            uvicorn_logger.handlers = root_logger.handlers
            uvicorn_logger.setLevel(getattr(logging, log_level))
            uvicorn_logger.propagate = False

    # 4. Structured JSON logger (Grafana) — separate logger, no propagation
    structured_logger = logging.getLogger("structured")
    structured_logger.setLevel(logging.INFO)
    structured_logger.propagate = False
    structured_logger.handlers.clear()

    if ENABLE_GRAFANA_LOGS:
        json_handler = logging.FileHandler('logs/api/structured.jsonl', encoding='utf-8')
        json_handler.setFormatter(logging.Formatter('%(message)s'))
        json_handler.setLevel(logging.INFO)
        json_handler.name = "json_handler"
        json_handler.addFilter(JSONOnlyFilter())  # Only emit valid JSON lines
        structured_logger.addHandler(json_handler)
        root_logger.info("Structured Grafana logs enabled")

    root_logger.info(f"Logging configured — level: {log_level}, Grafana: {ENABLE_GRAFANA_LOGS}")

    return root_logger


# Initialize logger and structured logger at module level
logger = setup_logging()
structured_logger = logging.getLogger("structured")

# ------------------------------------
# Persistence PostgreSQL Logging — JobStore 
# ------------------------------------
job_store = JobStore(JOBSTORE_DSN)

# ------------------------------------
# Helpers — structured logging & task tracking
# ------------------------------------

def emit_structured_log(payload: Dict[str, Any]) -> None:
    """
    Route a JSON event to the structured logger if Grafana logging is enabled.
    No-op if ENABLE_GRAFANA_LOGS is false.
    """
    if not ENABLE_GRAFANA_LOGS:
        return
    structured_logger.info(json.dumps(payload))


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
    """
    Update task status and metadata in ACTIVE_TASKS and JobStore (if enabled).

    Extra kwargs can include any additional tracking info, e.g.:
        current_rome, current_rome_label — for France Travail offer ingestion
        current_segment, current_url — for WTTJ ingestion
    """
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


# ============================================================
# SECTION 2 — Startup
# ============================================================

@app.on_event("startup")
def _startup_load_model():
    """
    Load ML model artifacts at startup.
    Avoids reloading MinIO on each prediction request.

    On failure: API continues without crashing (model endpoints return 503).
    Also marks stale jobs in JobStore (tasks that were RUNNING before restart).
    """
    global ARTIFACTS, rome_model
    logger.info("API starting — loading model artifacts...")
    try:
        ARTIFACTS = load_artifacts()
        logger.info(f"Model loaded successfully: {MODEL_NAME} v{ARTIFACTS['version']}")
        rome_model = get_rome_model()
        logger.info(f"ROME model loaded: {len(rome_model) if rome_model else 0} entries")

        # Emit structured startup event if Grafana logging is enabled
        if ENABLE_GRAFANA_LOGS:
            emit_structured_log({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "model_loaded",
                "model_name": MODEL_NAME,
                "version": ARTIFACTS['version']
            })

        # Mark as stale any jobs that were RUNNING before the last restart
        if job_store.enabled:
            stale_count = job_store.mark_stale(STALE_JOB_MINUTES)
            if stale_count:
                logger.warning("Marked stale jobs on startup: %s", stale_count)

    except Exception as e:
        ARTIFACTS = {}
        rome_model = {}
        logger.warning(
            "Model unavailable at startup (API continues without crashing): %s", e
        )


# ============================================================
# SECTION 3 — Background wrappers (tracking + business logic)
# ============================================================
# Each wrapper follows the same pattern:
#   1. Emit structured start event (if Grafana enabled)
#   2. Call log_to_db() for PostgreSQL trace
#   3. Call the underlying ingestion/processing function
#   4. On success: set_task(STATUS_SUCCESS) + job_store.finish() + emit structured end event
#   5. On failure: set_task(STATUS_FAILED) + job_store.finish() + emit structured failure event
#   6. On exception: same as failure path
#
# This ensures consistent tracking of task status, progress, and results across all background jobs,
# and use code reusability by separating the business logic (in the ingest/normalize functions) from the API exposition.

def run_normalize_wttj_task(task_id: str, dt: Optional[str], output_format: str) -> None:
    """Background wrapper for WTTJ normalization with task tracking."""
    start_monotonic = time.monotonic()
    try:
        result = normalize_wttj_jobs(dt, output_format)

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
                message="WTTJ normalization completed",
                records_count=len(result.files),
                errors_count=result.errors,
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )

            if job_store.enabled:
                job_store.finish(task_id, STATUS_SUCCESS, result=result_payload)
        else:
            error_text = f"WTTJ normalization failed: {result.status}"
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
            message=f"Error: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        if job_store.enabled:
            job_store.finish(task_id, STATUS_FAILED, error_text=error_text)


def run_normalize_ft_task(task_id: str, dt: Optional[str], output_format: str) -> None:
    """Background wrapper for France Travail normalization with task tracking."""
    start_monotonic = time.monotonic()
    try:
        
        result = normalize_ft_jobs(dt, output_format)

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
                message="FT normalization completed",
                records_count=len(result.files),
                errors_count=result.errors,
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )

            if job_store.enabled:
                job_store.finish(task_id, STATUS_SUCCESS, result=result_payload)
        else:
            error_text = f"FT normalization failed: {result.status}"
            
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
            message=f"Error: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        if job_store.enabled:
            job_store.finish(task_id, STATUS_FAILED, error_text=error_text)


def run_rome_metiers_task(task_id: str) -> None:
    """Background wrapper for ROME code ingestion with status updates."""
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
        log_to_db('rome_metiers', 'INFO', "Starting ROME code ingestion", task_id=task_id)
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
                'rome_metiers', 'INFO',
                f"{result.get('records_written')} ROME codes imported ({result.get('records_count')} total) - {duration_sec:.2f}s",
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
                message=result.get("message", "Ingestion failed"),
                records_count=result.get("records_count"),
                pages_count=result.get("calls", 0),
                errors_count=result.get("errors", 0),
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )

            log_to_db('rome_metiers', 'ERROR', f"Ingestion failed: {result.get('error')}", task_id=task_id, error=result.get('error'))
            
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
            message=f"Error: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )

        log_to_db('rome_metiers', 'ERROR', f"Exception after {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
        
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


def run_france_travail_offers_task(
    task_id: str,
    window_days: int,
    max_windows: int,
    binary_split_min_seconds: int,
    max_rome_codes: int
) -> None:
    """Background wrapper for France Travail offer ingestion with real-time progress tracking."""
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
        """Callback to update progress in real-time for each ROME code processed."""
        progress_pct = int((current / total) * 100) if total else 0
        set_task(
            task_id,
            progress_pct=progress_pct,
            message=f"Processing ROME code {current}/{total}: {rome_code} - {rome_label}",
            current_rome=rome_code,
            current_rome_label=rome_label,
        )

    try:
        log_to_db('france_travail_offers', 'INFO', "Starting France Travail offer ingestion", task_id=task_id)
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
                'france_travail_offers', 'INFO',
                (
                    f"{result.get('written')} offers imported ({result.get('rome_processed')} ROME codes, "
                    f"{result.get('calls')} calls, {result.get('errors')} errors) - {format_eta(duration_sec)}"
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
                message=result.get("message", "Ingestion failed"),
                records_count=result.get("written", 0),
                pages_count=result.get("calls", 0),
                errors_count=result.get("errors", 0),
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )
            log_to_db('france_travail_offers', 'ERROR', f"Ingestion failed: {result.get('error')}", task_id=task_id, error=result.get('error'))
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
            message=f"Error: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        log_to_db('france_travail_offers', 'ERROR', f"Exception after {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
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
    store_html_mode: str | None = "skip"
    
) -> None:
    """Background wrapper for Welcome to the Jungle ingestion with real-time progress tracking."""
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
        """Callback to update progress in real-time for each URL processed."""
        progress_pct = int((current / total) * 100) if total > 0 else 0
        set_task(
            task_id,
            progress_pct=progress_pct,
            message=f"Segment {segment}: {current}/{total} URLs processed (✓{ok} ✗{ko})",
            pages_count=current,
            errors_count=ko,
            current_segment=segment,
            current_url=current,
            total_urls=total,
        )

    try:
        log_to_db('welcome_to_the_jungle', 'INFO', "Starting Welcome to the Jungle ingestion", task_id=task_id)
        result = ingest_welcome_to_the_jungle(
            storage=None,
            mode=mode,
            max_jobs=max_jobs,
            max_companies=max_companies,
            workers=workers,
            part_size=part_size,
            provided_run_id=provided_run_id,
            progress_callback=update_progress,
            store_html_mode=store_html_mode
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
                'welcome_to_the_jungle', 'INFO',
                (
                    f"{result.get('total_written')} offers imported ({result.get('total_processed')} URLs, "
                    f"{errors_count} errors) - {format_eta(duration_sec)}"
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
                message=result.get("message", "Ingestion failed"),
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )
            log_to_db('welcome_to_the_jungle', 'ERROR', f"Ingestion failed: {result.get('error')}", task_id=task_id, error=result.get('error'))
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
            message=f"Error: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        log_to_db('welcome_to_the_jungle', 'ERROR', f"Exception after {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
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


def run_collect_sitemaps_task(task_id: str, delay: float, max_results: int) -> None:
    """Background wrapper for WTTJ sitemap URL collection with status updates."""
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
        log_to_db('wttj_sitemaps', 'INFO', "Starting WTTJ sitemap collection", task_id=task_id)
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
                message=f"Collection successful: {urls_count} URLs collected",
                records_count=urls_count,
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )
            log_to_db(
                'wttj_sitemaps', 'INFO',
                f"{urls_count} URLs collected - {format_eta(duration_sec)} (storage: {storage_key})",
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
            error_text = result.get("error", "Unknown error")
            set_task(
                task_id,
                message=f"Collection failed: {error_text}",
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )
            log_to_db('wttj_sitemaps', 'ERROR', f"Collection failed: {error_text}", task_id=task_id, error=error_text)
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
            message=f"Error: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        log_to_db('wttj_sitemaps', 'ERROR', f"Exception after {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
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
) -> None:
    """Background wrapper for optimized WTTJ job ingestion (REST API crawler) with status updates."""
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
        """Callback to update progress in real-time."""
        progress_pct = int((current / total) * 100) if total > 0 else 0
        set_task(
            task_id,
            progress_pct=progress_pct,
            message=f"Optimized ingestion: {current}/{total} URLs processed (✓{ok} ✗{ko})",
            pages_count=current,
            errors_count=ko,
        )

    try:
        log_to_db('wttj_opt', 'INFO', "Starting optimized WTTJ ingestion (REST API)", task_id=task_id)
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
                message=f"Optimized ingestion completed: {records_written} records written",
                records_count=records_written,
                pages_count=urls_processed,
                errors_count=urls_ko,
                status=STATUS_SUCCESS,
                completed_at=datetime.now(timezone.utc),
                result=result_payload,
            )
            log_to_db(
                'wttj_opt', 'INFO',
                f"{records_written} records written ({urls_processed} URLs, {urls_ko} errors) - {format_eta(duration_sec)}",
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
            error_text = result.get("error", "Unknown error")
            set_task(
                task_id,
                message=f"Optimized ingestion failed: {error_text}",
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )
            log_to_db('wttj_opt', 'ERROR', f"Ingestion failed: {error_text}", task_id=task_id, error=error_text)
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
            message=f"Error: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        log_to_db('wttj_opt', 'ERROR', f"Exception after {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
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


def run_merge_datasets_task(
    task_id: str,
    ft_prefix: Optional[str],
    wttj_prefix: Optional[str],
    output_prefix: Optional[str],
    output_format: str
) -> None:
    """Background wrapper for FT + WTTJ dataset merge with status updates."""
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
        """Callback to update progress in real-time for each merge step."""
        set_task(task_id, message=message, current_step=step)

    try:
        log_to_db('merge_datasets', 'INFO', "Starting FT + WTTJ dataset merge", task_id=task_id)
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
                'merge_datasets', 'INFO',
                f"Merge completed: {result.get('total_offers')} offers (FT: {result.get('ft_offers')}, WTTJ: {result.get('wttj_offers')}) - {duration_sec:.2f}s",
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
                message=result.get("message", "Merge failed"),
                status=STATUS_FAILED,
                completed_at=datetime.now(timezone.utc),
                error=error_text,
            )
            log_to_db('merge_datasets', 'ERROR', f"Merge failed: {result.get('error')}", task_id=task_id, error=error_text)
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
            message=f"Error: {error_text}",
            status=STATUS_FAILED,
            completed_at=datetime.now(timezone.utc),
            error=error_text,
        )
        log_to_db('merge_datasets', 'ERROR', f"Exception after {duration_sec:.2f}s: {e}", task_id=task_id, duration_sec=round(duration_sec, 2), error=error_text)
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


# ============================================================
# SECTION 4 — Monitoring endpoints
# ============================================================
# Read-only endpoints for health checks, task status, and job history.
# No side effects — safe to call at any time.

def _purge_completed_tasks(operations_filter: Optional[list] = None) -> None:
    """
    Remove completed tasks from ACTIVE_TASKS if they finished more than 5 minutes ago.
    Called at the start of each status endpoint to keep memory bounded.

    Args:
        operations_filter: if provided, only purge tasks matching these operation names.
    """
    current_time = datetime.now(timezone.utc)
    tasks_to_remove = [
        task_id
        for task_id, task_info in ACTIVE_TASKS.items()
        if task_info.get("status") == STATUS_SUCCESS
        and task_info.get("completed_at")
        and (current_time - task_info["completed_at"]).total_seconds() > 300
        and (operations_filter is None or task_info.get("operation") in operations_filter)
    ]
    for task_id in tasks_to_remove:
        del ACTIVE_TASKS[task_id]


@app.get(
    "/health",
    tags=["Monitoring"],
    summary="API health check",
    description="Verifies the API is operational and returns information about the loaded model."
)
def health():
    """Health check with model and storage backend information."""
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": ARTIFACTS.get("version"),
        "storage_backend": os.getenv("STORAGE_BACKEND", "local"),
    }


@app.get(
    "/jobs",
    tags=["Monitoring"],
    summary="List job runs",
    description="Returns runs persisted in JobStore."
)
def list_jobs(
    source: Optional[str] = Query(None, description="Filter by source"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results")
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
    summary="Job run detail",
    description="Returns a persistent run by run_id."
)
def get_job(run_id: str):
    if not job_store.enabled:
        raise HTTPException(status_code=503, detail="JobStore disabled")
    job = job_store.get_run(run_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {run_id} not found")
    return job


@app.get(
    "/tasks/{task_id}",
    tags=["Monitoring"],
    summary="Task details",
    description="Returns detailed information for a task (ingestion, merge, etc.)."
)
def get_task_details_generic(task_id: str):
    """
    Returns full details of an asynchronous task.

    Possible statuses:
        - RUNNING: task is in progress
        - SUCCESS: task completed successfully
        - FAILED: task failed
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
    "/ingest/status",
    tags=["Monitoring"],
    summary="Ingestion operations status",
    description="Lists all available ingestion operations and currently running tasks."
)
def get_ingest_status():
    """Returns available ingestion operations and active tasks. Auto-purges tasks completed 5+ minutes ago."""
    ingestion_operations = [
        "ingest_rome_metiers",
        "ingest_france_travail_offers",
        "ingest_welcome_to_jungle",
        "collect_wttj_sitemaps",
        "ingest_wttj_opt"
    ]
    _purge_completed_tasks(ingestion_operations)
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
            {"endpoint": "POST /ingest/rome-metiers", "description": "Ingest ROME job codes from France Travail", "params": ["background (bool, optional)"]},
            {"endpoint": "POST /ingest/france-travail-offers", "description": "Ingest France Travail job offers (bronze layer)", "params": ["background", "max_rome_codes", "window_days"]},
            {"endpoint": "POST /ingest/welcome-to-jungle", "description": "Ingest Welcome to the Jungle job offers (bronze layer)", "params": ["background", "mode (new/resume/incremental)", "max_jobs", "max_companies"]},
            {"endpoint": "POST /ingest/welcome-to-the-jungle/sitemaps", "description": "Collect WTTJ URLs from XML sitemaps", "params": ["background", "delay", "max_results"]},
            {"endpoint": "POST /ingest/welcome-to-the-jungle/jobs-optimized", "description": "Optimized WTTJ ingestion via REST API with JSON-LD fallback", "params": ["background", "mode", "max_urls", "workers", "part_size", "delay", "force_download_urls"]},
        ]
    }


@app.get(
    "/data/status",
    tags=["Monitoring"],
    summary="Data processing operations status",
    description="Lists all available data processing operations and currently running tasks."
)
def get_data_status():
    """Returns available data processing operations and active tasks. Auto-purges tasks completed 5+ minutes ago."""
    data_operations = ["merge_datasets", "normalize_wttj_jobs", "normalize_ft_jobs"]
    _purge_completed_tasks(data_operations)
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
            {"endpoint": "POST /data/normalize-ft-jobs", "description": "Normalize FT bronze to silver (canonical schema)", "params": ["background", "dt (YYYY-MM-DD/latest)", "output_format (parquet/jsonl/csv)"]},
            {"endpoint": "POST /data/normalize-wttj-jobs", "description": "Normalize WTTJ bronze to silver (with ROME enrichment)", "params": ["background", "dt (YYYY-MM-DD/latest)", "output_format (parquet/jsonl/csv)"]},
            {"endpoint": "POST /data/merge-datasets", "description": "Merge FT and WTTJ datasets with ROME codes", "params": ["background", "ft_prefix", "wttj_prefix", "output_prefix", "output_format (parquet/jsonl/csv)"]},
        ]
    }


# ============================================================
# SECTION 6 — Business endpoints
# ============================================================
# Each endpoint supports both synchronous and asynchronous execution:
#   - background=false (default): waits for completion, returns full result
#   - background=true: launches task in background, returns task_id immediately
#                      use GET /tasks/{task_id} to poll status

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Machine Learning"],
    summary="Predict the ROME code for a job offer",
    description="""Automatically classifies a job offer using the ROME framework.

Provide at minimum a job title or description.
Returns the most probable ROME code and a top-K ranking of relevant codes.
"""
)
def predict(req: PredictRequest):
    """
    ROME classification for a job offer.

    Args:
        intitule: Job title (optional but recommended)
        description: Detailed description (optional but recommended)
        competences: List of technical skills (optional)
    """
    logger.info(f"Prediction request — title: {req.intitule[:50] if req.intitule else 'N/A'}")

    if not ARTIFACTS:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts are not loaded yet. "
                "Run training or upload model artifacts, then restart the API."
            ),
        )

    text = build_text_payload(
        intitule=req.intitule,
        description=req.description,
        competences=req.competences,
    )
    pred = predict_top_k(ARTIFACTS, text, top_k=TOP_K, rome_index=rome_model)
    logger.info(f"Prediction successful — ROME: {pred.get('rome_pred', 'N/A')} / {pred.get('rome_label', 'N/A')}")
    return pred


@app.post(
    "/ingest/rome-metiers",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingest ROME job codes",
    description="""Triggers ingestion of the complete ROME job nomenclature from the France Travail API.

Data is stored in `bronze/rome/rome_metiers.jsonl` (~532 ROME codes).

**Execution modes:**
- **Synchronous** (background=false): waits for completion, returns result
- **Asynchronous** (background=true): starts background task, returns immediately
"""
)
async def ingest_rome_metiers_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Run in background")
):
    logger.info(f"ROME code ingestion request (background={background})")

    if background:
        task_id = utc_run_id()
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_rome_metiers",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": "Ingesting ROME job codes..."
        }
        if job_store.enabled:
            job_store.create(run_id=task_id, job_type="import", source="rome_metiers", params={"background": True}, message="Ingesting ROME job codes...")
        background_tasks.add_task(run_rome_metiers_task, task_id)
        return IngestResponse(success=True, message=f"ROME code ingestion started in background (task_id: {task_id})", key=task_id)
    else:
        try:
            result = ingest_rome_metiers()
            if result["success"]:
                logger.info(f"Ingestion successful: {result['records_count']} ROME codes")
            else:
                logger.error(f"Ingestion failed: {result.get('error')}")
            return IngestResponse(**result)
        except Exception as e:
            logger.error(f"Ingestion error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.post(
    "/ingest/france-travail-offers",
    response_model=IngestOffersResponse,
    tags=["Ingestion"],
    summary="Ingest France Travail job offers",
    description="""Triggers complete ingestion of France Travail job offers into the bronze layer.

This operation can take several hours for all ROME codes.
Asynchronous mode is recommended.

**Control parameters:**
- **max_rome_codes**: limit ROME codes to process (0 = all, useful for testing)
- **window_days**: time window size in days (default: 7)
- **max_windows**: maximum number of time windows (default: 260)
"""
)
async def ingest_france_travail_offers_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Run in background"),
    max_rome_codes: int = Query(0, description="Limit ROME codes (0 = all)"),
    window_days: int = Query(7, description="Time window size in days"),
    max_windows: int = Query(260, description="Maximum number of time windows"),
    binary_split_min_seconds: int = Query(3600, description="Minimum window size for binary split")
):
    logger.info(f"FT offer ingestion request (background={background}, max_rome_codes={max_rome_codes})")

    if background:
        task_id = utc_run_id()
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_france_travail_offers",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Ingesting France Travail offers (max_rome_codes: {max_rome_codes or 'all'})...",
            "params": {"max_rome_codes": max_rome_codes, "window_days": window_days, "max_windows": max_windows}
        }
        if job_store.enabled:
            job_store.create(run_id=task_id, job_type="import", source="france_travail_offers", params={"background": True, "max_rome_codes": max_rome_codes, "window_days": window_days, "max_windows": max_windows, "binary_split_min_seconds": binary_split_min_seconds}, message=f"Ingesting France Travail offers (max_rome_codes: {max_rome_codes or 'all'})...")
        background_tasks.add_task(run_france_travail_offers_task, task_id, window_days, max_windows, binary_split_min_seconds, max_rome_codes)
        return IngestOffersResponse(success=True, message=f"France Travail offer ingestion started in background (task_id: {task_id})", run_id=task_id)
    else:
        try:
            result = ingest_france_travail_offers(storage=None, client=None, window_days=window_days, max_windows=max_windows, binary_split_min_seconds=binary_split_min_seconds, max_rome_codes=max_rome_codes, logger_override=None)
            if result["success"]:
                logger.info(f"Ingestion successful: {result['written']} offers, {result['rome_processed']} ROME codes")
            else:
                logger.error(f"Ingestion failed: {result.get('error')}")
            return IngestOffersResponse(**result)
        except Exception as e:
            logger.error(f"Ingestion error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.post(
    "/ingest/welcome-to-jungle",
    response_model=IngestWTTJResponse,
    tags=["Ingestion"],
    summary="Ingest Welcome to the Jungle job offers",
    description="""Triggers ingestion of Welcome to the Jungle job offers into the bronze layer.

Collects URLs from sitemaps and extracts job and company page data.

**Ingestion modes:**
- **new**: fresh run with generated run_id, no skip
- **resume**: resumes an existing run (requires run_id)
- **incremental**: skips URLs already processed in a previous run

**store_html_mode** controls how HTML content is handled:
- **skip** (default): do not store HTML content
- **store**: store full HTML content in the dataset as fallback for reprocessing if needed (increases storage usage)
- **store_links**: store only links to the HTML content in the dataset (HTML stored separately in object storage, reduces dataset size but requires additional fetch to access HTML)    
"""
)
async def ingest_wttj_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Run in background"),
    mode: str = Query("new", description="Ingestion mode (new, resume, incremental)"),
    max_jobs: int = Query(0, description="Limit number of jobs (0 = all)"),
    max_companies: int = Query(0, description="Limit number of companies (0 = all)"),
    workers: int = Query(6, description="Number of concurrent workers"),
    part_size: int = Query( 1000, description="JSONL chunk size in records"),
    provided_run_id: str | None = Query(None, description="Run ID to use in resume mode"),
    store_html_mode: str = Query("skip", description="How to handle HTML content (skip, store, store_links)")
    
):
    logger.info("WTTJ ingestion request (background=%s, mode=%s, max_jobs=%s, max_companies=%s, workers=%s, part_size=%s)", background, mode, max_jobs, max_companies, workers, part_size)

    if background:
        task_id = utc_run_id()
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_welcome_to_jungle",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"WTTJ ingestion in progress (mode: {mode}, jobs: {max_jobs or 'all'}, companies: {max_companies or 'all'}, workers: {workers})...",
            "params": {"mode": mode, 
                       "max_jobs": max_jobs, 
                       "max_companies": max_companies, 
                       "workers": workers, 
                       "part_size": part_size, 
                       "provided_run_id": provided_run_id, 
                       }
        }
        if job_store.enabled:
            job_store.create(run_id=task_id, 
                             job_type="import", 
                             source="wttj", 
                             params={"background": True, 
                                     "mode": mode, 
                                     "max_jobs": max_jobs, 
                                     "max_companies": max_companies, 
                                     "workers": workers, 
                                     "part_size": part_size, 
                                     "provided_run_id": provided_run_id, 
                                     "store_html_mode": store_html_mode
                                     },
                            message=f"WTTJ ingestion in progress (mode: {mode})..."
                            )
            
        background_tasks.add_task(run_welcome_to_jungle_task, 
                                  task_id, 
                                  mode, 
                                  max_jobs, 
                                  max_companies, 
                                  workers, 
                                  part_size, 
                                  provided_run_id or task_id,
                                  store_html_mode
                                  )
        
        return IngestWTTJResponse(success=True, message=f"WTTJ ingestion started in background (task_id: {task_id})", run_id=task_id)
    else:
        try:
            result = ingest_welcome_to_the_jungle(
                storage=None, 
                mode=mode, 
                max_jobs=max_jobs, 
                max_companies=max_companies, 
                workers=workers, 
                part_size=part_size, 
                provided_run_id=provided_run_id 
            )
            if result["success"]:
                logger.info(f"Ingestion successful: {result.get('total_written')} records")
            else:
                logger.error(f"Ingestion failed: {result.get('error')}")
            return IngestWTTJResponse(**result)
        except Exception as e:
            logger.error(f"WTTJ ingestion error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.post(
    "/ingest/welcome-to-the-jungle/sitemaps",
    response_model=CollectSitemapsResponse,
    tags=["Ingestion"],
    summary="Collect URLs from WTTJ sitemaps",
    description="""Collects job page URLs from Welcome to the Jungle XML sitemaps.

Stores collected URLs in the bronze storage for later processing.
This operation is typically fast (~seconds for ~10k URLs).
"""
)
async def collect_sitemaps_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Run in background"),
    delay: float = Query(0.5, description="Delay between requests (seconds)"),
    max_results: int = Query(0, description="Max URLs to collect (0 = all)")
):
    logger.info(f"WTTJ sitemap collection request (background={background}, delay={delay}, max_results={max_results})")

    if background:
        task_id = utc_run_id()
        ACTIVE_TASKS[task_id] = {
            "operation": "collect_wttj_sitemaps",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Collecting WTTJ sitemaps (max_results: {max_results or 'all'}, delay: {delay}s)...",
            "params": {"delay": delay, "max_results": max_results}
        }
        if job_store.enabled:
            job_store.create(run_id=task_id, 
                             job_type="import", 
                             source="wttj_sitemaps", 
                             params={"background": True, 
                                     "delay": delay, 
                                     "max_results": max_results}, 
                            message=f"Collecting WTTJ sitemaps...")
            
        background_tasks.add_task(run_collect_sitemaps_task, task_id, delay, max_results)
        return CollectSitemapsResponse(success=True, message=f"WTTJ sitemap collection started in background (task_id: {task_id})", 
                                       urls_count=0, 
                                       storage_key=task_id, 
                                       elapsed_s=None, 
                                       error=None)
    else:
        try:
            start_time = time.time()
            result = collect_sitemap_urls(query="", entreprise="", ville="", max_results=max_results, delay=delay)
            elapsed_s = time.time() - start_time

            if result.get("success"):
                urls_count = result.get("total_processed", 0)
                storage_key = result.get("storage_key")
                logger.info(f"Collection successful: {urls_count} URLs in {elapsed_s:.2f}s (storage: {storage_key})")
                return CollectSitemapsResponse(success=True, 
                                               message=f"Collection successful: {urls_count} URLs collected", 
                                               urls_count=urls_count, 
                                               storage_key=storage_key, 
                                               elapsed_s=elapsed_s, 
                                               error=None)
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Collection failed: {error_msg}")
                return CollectSitemapsResponse(success=False, 
                                               message=f"Collection failed: {error_msg}", 
                                               urls_count=0, 
                                               storage_key=None, 
                                               elapsed_s=elapsed_s, 
                                               error=error_msg)
        except Exception as e:
            logger.error(f"Sitemap collection error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Sitemap collection error: {str(e)}")


@app.post(
    "/ingest/welcome-to-the-jungle/jobs-optimized",
    response_model=IngestWTTJOptResponse,
    tags=["Ingestion"],
    summary="Ingest WTTJ jobs via optimized crawler",
    description="""Triggers optimized Welcome to the Jungle job ingestion via REST API crawler.

**Ingestion modes:** new / resume / incremental

**Control parameters:**
- **max_urls**: limit URLs to process (0 = all)
- **workers**: concurrent workers (default: 8)
- **part_size**: JSONL chunk size (default: 5000)
- **delay**: per-thread request delay in seconds (default: 2.0)
- **force_download_urls**: force re-download of URLs from sitemaps
"""
)
async def ingest_wttj_jobs_optimized_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Run in background"),
    mode: str = Query("new", description="Ingestion mode (new, resume, incremental)"),
    max_urls: int = Query(0, description="Limit URLs to process (0 = all)"),
    workers: int = Query(8, description="Number of concurrent workers"),
    part_size: int = Query(5000, description="JSONL chunk size in records"),
    delay: float = Query(2.0, description="Per-thread request delay (seconds)"),
    force_download_urls: bool = Query(True, description="Force re-download of URLs from sitemaps")
):
    logger.info(f"Optimized WTTJ ingestion request (background={background}, mode={mode}, max_urls={max_urls}, workers={workers}, delay={delay})")

    if background:
        task_id = utc_run_id()
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_wttj_opt",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Optimized WTTJ ingestion in progress (mode: {mode}, max_urls: {max_urls or 'all'}, workers: {workers})...",
            "params": {"mode": mode, "max_urls": max_urls, "workers": workers, "part_size": part_size, "delay": delay, "force_download_urls": force_download_urls}
        }
        if job_store.enabled:
            job_store.create(run_id=task_id, 
                             job_type="import", 
                             source="wttj_opt", 
                             params={"background": True, 
                                     "mode": mode, 
                                     "max_urls": max_urls, 
                                     "workers": workers, 
                                     "part_size": part_size, 
                                     "delay": delay, 
                                     "force_download_urls": force_download_urls
                                     }, 
                            message=f"Optimized WTTJ ingestion in progress..."
                            )
            
        background_tasks.add_task(run_wttj_job_opt_task, 
                                  task_id, 
                                  mode, 
                                  max_urls, 
                                  workers, 
                                  part_size, 
                                  delay, 
                                  force_download_urls)
        
        return IngestWTTJOptResponse(success=True, 
                                     message=f"Optimized WTTJ ingestion started in background (task_id: {task_id})", 
                                     run_id=task_id, 
                                     mode=mode, 
                                     urls_total=None, 
                                     urls_processed=None, 
                                     urls_ok=None, 
                                     urls_ko=None, 
                                     records_written=None, 
                                     elapsed_s=None, 
                                     error=None)
    else:
        try:
            result = ingest_welcome_to_the_jungle_opt(storage=None, 
                                                      mode=mode, 
                                                      max_urls=max_urls, 
                                                      workers=workers, 
                                                      part_size=part_size, 
                                                      delay=delay, 
                                                      provided_run_id=None, 
                                                      progress_callback=None, 
                                                      force_download_urls=force_download_urls)
            if result.get("success"):
                jobs_opt = result.get("jobs_opt", {})
                urls_processed = jobs_opt.get("processed", 0)
                urls_ok = jobs_opt.get("ok", 0)
                urls_ko = jobs_opt.get("ko", 0)
                records_written = jobs_opt.get("written", 0)
                logger.info(f"Ingestion successful: {records_written} records ({urls_processed} URLs, {urls_ko} errors)")
                return IngestWTTJOptResponse(success=True, 
                                             message=result.get("message", ""), 
                                             run_id=result.get("run_id"), 
                                             dt=result.get("dt"), 
                                             mode=result.get("mode"), 
                                             urls_total=urls_processed, 
                                             urls_processed=urls_processed, 
                                             urls_ok=urls_ok, 
                                             urls_ko=urls_ko, 
                                             records_written=records_written, 
                                             elapsed_s=result.get("elapsed_s"), 
                                             storage_prefix=f"dt={result.get('dt')}/run_id={result.get('run_id')}", 
                                             error=None)
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Ingestion failed: {error_msg}")
                return IngestWTTJOptResponse(success=False, 
                                             message=result.get("message", "Ingestion failed"), 
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
                                             error=error_msg)
        except Exception as e:
            logger.error(f"Optimized WTTJ ingestion error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.post(
    "/data/normalize-wttj-jobs",
    response_model=NormalizeWTTJResponse,
    tags=["Data Processing"],
    summary="Normalize WTTJ bronze jobs data to a silver format and enrich it with predicted ROME codes",
    description="Reads all bronze job_raw files for a given dt, clean and normalize data, enriches it with predicted ROME codes and save it in a silver format with the same dt. " \
    "Write silver layer preserving directory structure. " \
    "" \
    ""
)
async def normalize_wttj_jobs_endpoint(
    background_tasks: BackgroundTasks,
    dt: Optional[str] = Query(None, description="Extraction date in bronze layer (YYYY-MM-DD). If not provided or 'latest', uses the most recent available dt."),
    output_format: str = Query(default="parquet", description="Output format: parquet (default), jsonl, or csv"),
    background: bool = Query(default=True, description="Run task in background")
):
    if background:
        task_id = utc_run_id()
        ACTIVE_TASKS[task_id] = {
            "operation": "normalize_wttj_jobs",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"WTTJ normalization in progress (format: {output_format})...",
            "params": {"dt": dt, "output_format": output_format},
        }
        if job_store.enabled:
            job_store.create(run_id=task_id,
                             job_type="data",
                             source="normalize_wttj_jobs",
                             params={"background": True,
                                     "dt": dt,
                                     "output_format": output_format
                                     },
                            message=f"WTTJ normalization in progress (format: {output_format})..."
                            )

        background_tasks.add_task(run_normalize_wttj_task, task_id, dt, output_format)
        return NormalizeWTTJResponse(job_id=task_id, status="RUNNING", dt=dt, format=output_format, files=[], errors=0)
    else:
        r = normalize_wttj_jobs(dt, output_format)
        return NormalizeWTTJResponse(job_id=r.job_id, status=r.status, dt=r.dt, format=r.format, files=r.files, errors=r.errors)


@app.post(
    "/data/normalize-ft-jobs",
    response_model=NormalizeFTResponse,
    tags=["Data Processing"],
    summary="Normalize FT bronze jobs data to a silver format",
    description="Reads all FT bronze job_raw files for a given dt, clean and normalize data and save it in a silver format with the same dt."
)
async def normalize_ft_jobs_endpoint(
    background_tasks: BackgroundTasks,
    dt: Optional[str] = Query(None, description="Extraction date in bronze layer (YYYY-MM-DD). If not provided or 'latest', uses the most recent available dt."),
    output_format: str = Query(default="parquet", description="Output format: parquet (default), jsonl, or csv"),
    background: bool = Query(default=True, description="Run task in background")
):
    if background:
        task_id = utc_run_id()

        ACTIVE_TASKS[task_id] = {
            "operation": "normalize_ft_jobs",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"FT normalization in progress (format: {output_format})...",
            "params": {"dt": dt, "output_format": output_format},
        }

        if job_store.enabled:

            job_store.create(run_id=task_id,
                             job_type="data",
                             source="normalize_ft_jobs",
                             params={"background": True,
                                     "dt": dt,
                                     "output_format": output_format
                                     },
                            message=f"FT normalization in progress (format: {output_format})..."
                            )

        background_tasks.add_task(run_normalize_ft_task, task_id, dt, output_format)

        return NormalizeFTResponse(job_id=task_id, status="RUNNING", dt=dt, format=output_format, files=[], errors=0)
    else:
        r = normalize_ft_jobs(dt, output_format)
        return NormalizeFTResponse(job_id=r.job_id, status=r.status, dt=r.dt, format=r.format, files=r.files, errors=r.errors)


@app.post(
    "/data/merge-datasets",
    response_model=MergeDatasetResponse,
    tags=["Data Processing"],
    summary="Merge FT and WTTJ datasets",
    description="""Triggers the merge of France Travail and Welcome to the Jungle datasets.

Reads already-normalized Silver data (FT and WTTJ), merges and deduplicates
to create a unified training dataset.

**Steps:**
1. Auto-detect prefixes if not specified
2. Read FT Silver data
3. Read WTTJ Silver data
4. Harmonize canonical schema
5. Merge and deduplicate by URL
6. Compute statistics
7. Save merged dataset

**Output formats:** parquet (recommended), jsonl, csv
"""
)
async def merge_datasets_endpoint(
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Run in background"),
    ft_prefix: Optional[str] = Query(None, description="FT data prefix (auto-detect if not specified)"),
    wttj_prefix: Optional[str] = Query(None, description="WTTJ data prefix (auto-detect if not specified)"),
    output_prefix: Optional[str] = Query(None, description="Output prefix (default: datasets/ft_wttj_merged)"),
    output_format: str = Query("parquet", description="Output format (parquet, jsonl, csv)")
):
    logger.info(f"Dataset merge request (background={background}, format={output_format})")

    if background:
        task_id = utc_run_id()
        ACTIVE_TASKS[task_id] = {
            "operation": "merge_datasets",
            "status": STATUS_RUNNING,
            "started_at": datetime.now(timezone.utc),
            "progress_pct": 0,
            "message": f"Merging datasets (format: {output_format})...",
            "params": {"ft_prefix": ft_prefix, "wttj_prefix": wttj_prefix, "output_prefix": output_prefix, "output_format": output_format}
        }
        if job_store.enabled:
            job_store.create(run_id=task_id,
                             job_type="data",
                             source="merge",
                             params={"background": True,
                                     "ft_prefix": ft_prefix,
                                     "wttj_prefix": wttj_prefix,
                                     "output_prefix": output_prefix,
                                     "output_format": output_format}, message=f"Merging datasets (format: {output_format})..."
                            )

        background_tasks.add_task(run_merge_datasets_task,
                                  task_id,
                                  ft_prefix,
                                  wttj_prefix,
                                  output_prefix,
                                  output_format)

        return MergeDatasetResponse(success=True, message=f"Dataset merge started in background (task_id: {task_id})", output_key=task_id)
    else:
        try:
            result = merge_ft_wttj_datasets(ft_prefix=ft_prefix,
                                            wttj_prefix=wttj_prefix,
                                            output_prefix=output_prefix,
                                            output_format=output_format)
            if result["success"]:
                logger.info(f"Merge successful: {result.get('total_offers')} offers merged")
            else:
                logger.error(f"Merge failed: {result.get('error')}")
            return MergeDatasetResponse(**result)
        except Exception as e:
            logger.error(f"Merge error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Merge error: {str(e)}")