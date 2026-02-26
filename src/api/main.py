import os
import logging
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query

from src.models.predict_model import build_text_payload, load_artifacts, predict_top_k, get_rome_model
from src.ingest.bronze.france_travail_rome_metiers import ingest_rome_metiers
from src.ingest.bronze.france_travail import ingest_france_travail_offers
from src.ingest.bronze.welcome_to_the_jungle import ingest_welcome_to_the_jungle
from src.data.make_merge_dataset_ft_wttj_with_rome import merge_ft_wttj_datasets

ENABLE_GRAFANA_LOGS = os.getenv("ENABLE_GRAFANA_LOGS", "false").lower() == "true"

# Import des modèles API
from src.api.models import (
    PredictRequest,
    PredictResponse,
    IngestResponse,
    IngestOffersResponse,
    IngestWTTJResponse,
    MergeDatasetResponse
)

# Filtre pour logs JSON uniquement (utilisé pour structured.jsonl)
class JSONOnlyFilter(logging.Filter):
    """Filtre qui ne laisse passer que les messages JSON"""
    def filter(self, record):
        # Ne garder que les messages qui ressemblent à du JSON
        return record.getMessage().strip().startswith('{')

# Configuration du logging avec rotation et support optionnel Grafana
def setup_logging():
    """Configure le logging avec rotation de fichiers et logs structurés optionnels"""
    # Variables d'environnement pour la configuration
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10*1024*1024))  # 10MB par défaut
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Créer la structure de dossiers logs
    Path("logs/api").mkdir(parents=True, exist_ok=True)
    Path("logs/ingestion").mkdir(parents=True, exist_ok=True)
    Path("logs/prediction").mkdir(parents=True, exist_ok=True)
    
    # Logger racine
    root_logger = logging.getLogger(__name__)
    root_logger.setLevel(getattr(logging, log_level))
    
    # Formatter standard (lisible)
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(standard_formatter)
    console_handler.setLevel(getattr(logging, log_level))
    
    # 2. Fichier principal de l'API
    file_handler = RotatingFileHandler(
        'logs/api/main.log',
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(standard_formatter)
    file_handler.setLevel(getattr(logging, log_level))
    
    # 3. Fichier d'erreurs global
    error_handler = RotatingFileHandler(
        'logs/api/errors.log',
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
        encoding='utf-8'
    )
    error_handler.setFormatter(standard_formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Ajouter les handlers de base
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # 4. Logger dédié aux événements structurés (JSON uniquement)
    structured_logger = logging.getLogger("structured")
    structured_logger.setLevel(logging.INFO)
    structured_logger.propagate = False
    structured_logger.handlers.clear()

    if ENABLE_GRAFANA_LOGS:
        json_handler = logging.FileHandler('logs/api/structured.jsonl', encoding='utf-8')
        json_handler.setFormatter(logging.Formatter('%(message)s'))
        json_handler.setLevel(logging.INFO)
        json_handler.name = "json_handler"
        json_handler.addFilter(JSONOnlyFilter())  # Filtre JSON uniquement
        structured_logger.addHandler(json_handler)
        root_logger.info("📊 Logs structurés Grafana activés")
    
    root_logger.info(f"📝 Logging configuré - Niveau: {log_level}, Grafana: {ENABLE_GRAFANA_LOGS}")
    
    return root_logger


def get_endpoint_logger(endpoint_name: str, category: str = "api") -> logging.Logger:
    """Crée un logger spécifique pour un endpoint avec son propre fichier de log
    
    Args:
        endpoint_name: Nom de l'endpoint (ex: 'rome_metiers', 'prediction', 'wttj')
        category: Catégorie du log ('api', 'ingestion', 'prediction')
    
    Returns:
        Logger configuré pour cet endpoint
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10*1024*1024))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Créer un logger unique pour cet endpoint
    logger_name = f"{__name__}.{category}.{endpoint_name}"
    endpoint_logger = logging.getLogger(logger_name)
    endpoint_logger.setLevel(getattr(logging, log_level))
    
    # Éviter la duplication si déjà configuré
    if endpoint_logger.handlers:
        return endpoint_logger
    
    # Formatter standard
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler fichier spécifique à l'endpoint
    log_path = f'logs/{category}/{endpoint_name}.log'
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(standard_formatter)
    file_handler.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(standard_formatter)
    console_handler.setLevel(getattr(logging, log_level))
    
    endpoint_logger.addHandler(file_handler)
    endpoint_logger.addHandler(console_handler)
    
    # Éviter la propagation pour ne pas dupliquer dans les logs parents
    endpoint_logger.propagate = False
    
    return endpoint_logger

logger = setup_logging()
structured_logger = logging.getLogger("structured")


def emit_structured_log(payload: Dict[str, Any]) -> None:
    """Route un événement JSON vers le logger structuré si Grafana est activé."""
    if not ENABLE_GRAFANA_LOGS:
        return
    structured_logger.info(json.dumps(payload))

MODEL_NAME = os.getenv("MODEL_NAME", "rome_tfidf")
TOP_K = int(os.getenv("TOP_K", "5"))

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

# Cache global (chargé une seule fois)
ARTIFACTS: Dict[str, Any] = {}

# Tracking des tâches d'ingestion en cours
ACTIVE_TASKS: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
def _startup_load_model():
    """
    Charge les artefacts une seule fois au démarrage.
    Evite de recharger MinIO/joblib à chaque requête.
    """
    global ARTIFACTS
    logger.info("🚀 Démarrage de l'API - Chargement du modèle...")
    try:
        ARTIFACTS = load_artifacts()
        logger.info(f"✅ Modèle chargé avec succès: {MODEL_NAME} v{ARTIFACTS['version']}")
        
        # Log structuré si Grafana activé
        if ENABLE_GRAFANA_LOGS:
            structured_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "model_loaded",
                "model_name": MODEL_NAME,
                "version": ARTIFACTS['version']
            }
            emit_structured_log(structured_log)
                    
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {e}", exc_info=True)
        raise


# =====================================
# Wrappers pour tâches d'ingestion avec tracking
# =====================================

def run_rome_metiers_task(task_id: str):
    """Wrapper pour l'ingestion des codes ROME avec mise à jour du statut"""
    import time
    
    # Logger spécifique pour cette ingestion
    task_logger = get_endpoint_logger('rome_metiers', 'ingestion')
    
    start_time = time.time()
    
    try:
        task_logger.info(f"[{task_id}] 📥 Début de l'ingestion des codes ROME")
        result = ingest_rome_metiers()
        
        duration_sec = time.time() - start_time
        
        if result["success"]:
            ACTIVE_TASKS[task_id].update({
                "status": "completed",
                "progress": "100%",
                "message": result["message"],
                "completed_at": datetime.now(),
                "result": {
                    "records_count": result.get("records_count"),
                    "records_written": result.get("records_written"),
                    "key": result.get("key"),
                    "duration_sec": round(duration_sec, 2)
                }
            })
            task_logger.info(
                f"[{task_id}] ✅ Ingestion terminée avec succès - "
                f"{result.get('records_count')} codes ROME en {duration_sec:.2f}s"
            )
            
            # Log structuré si Grafana activé
            if ENABLE_GRAFANA_LOGS:
                structured_log = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "ingestion_completed",
                    "task_id": task_id,
                    "task_type": "rome_metiers",
                    "status": "success",
                    "records_count": result.get("records_count"),
                    "duration_sec": round(duration_sec, 2)
                }
                emit_structured_log(structured_log)
        else:
            ACTIVE_TASKS[task_id].update({
                "status": "failed",
                "progress": "N/A",
                "message": result.get("message", "Échec de l'ingestion"),
                "completed_at": datetime.now(),
                "error": result.get("error")
            })
            task_logger.error(f"[{task_id}] ❌ Échec de l'ingestion: {result.get('error')}")
            
            # Log structuré si Grafana activé
            if ENABLE_GRAFANA_LOGS:
                structured_log = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "ingestion_failed",
                    "task_id": task_id,
                    "task_type": "rome_metiers",
                    "status": "failed",
                    "error": result.get("error"),
                    "duration_sec": round(duration_sec, 2)
                }
                emit_structured_log(structured_log)
                        
    except Exception as e:
        duration_sec = time.time() - start_time
        ACTIVE_TASKS[task_id].update({
            "status": "failed",
            "progress": "N/A",
            "message": f"Erreur: {str(e)}",
            "completed_at": datetime.now(),
            "error": str(e)
        })
        task_logger.error(
            f"[{task_id}] ❌ Exception après {duration_sec:.2f}s: {e}", 
            exc_info=True
        )
        
        # Log structuré si Grafana activé
        if ENABLE_GRAFANA_LOGS:
            structured_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "ingestion_error",
                "task_id": task_id,
                "task_type": "rome_metiers",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_sec": round(duration_sec, 2)
            }
            emit_structured_log(structured_log)


def run_france_travail_offers_task(task_id: str, window_days: int, max_windows: int, 
                                     binary_split_min_seconds: int, max_rome_codes: int):
    """Wrapper pour l'ingestion des offres FT avec mise à jour du statut"""
    import time
    
    # Logger spécifique pour cette ingestion
    task_logger = get_endpoint_logger('france_travail_offers', 'ingestion')
    
    start_time = time.time()
    
    def update_progress(current: int, total: int, rome_code: str, rome_label: str):
        """Callback pour mettre à jour la progression en temps réel"""
        progress_pct = int((current / total) * 100)
        ACTIVE_TASKS[task_id].update({
            "progress": f"{progress_pct}%",
            "message": f"Traitement code ROME {current}/{total}: {rome_code} - {rome_label}",
            "current_rome": rome_code,
            "current_rome_label": rome_label
        })
    
    try:
        task_logger.info(f"[{task_id}] 📥 Début de l'ingestion des offres France Travail")
        result = ingest_france_travail_offers(
            storage=None,
            client=None,
            window_days=window_days,
            max_windows=max_windows,
            binary_split_min_seconds=binary_split_min_seconds,
            max_rome_codes=max_rome_codes,
            progress_callback=update_progress,
            logger_override=task_logger,
            task_id=task_id,
        )
        
        duration_sec = time.time() - start_time
        
        if result["success"]:
            ACTIVE_TASKS[task_id].update({
                "status": "completed",
                "progress": "100%",
                "message": result["message"],
                "completed_at": datetime.now(),
                "result": {
                    "run_id": result.get("run_id"),
                    "run_key": result.get("run_key"),
                    "rome_processed": result.get("rome_processed"),
                    "calls": result.get("calls"),
                    "written": result.get("written"),
                    "elapsed_s": result.get("elapsed_s"),
                    "errors": result.get("errors"),
                    "duration_sec": round(duration_sec, 2)
                }
            })
            task_logger.info(
                f"[{task_id}] ✅ Ingestion terminée: {result.get('written')} offres en {duration_sec:.2f}s"
            )
            
            # Log structuré si Grafana activé - via ROOT logger
            if ENABLE_GRAFANA_LOGS:
                structured_log = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "ingestion_completed",
                    "task_id": task_id,
                    "task_type": "france_travail_offers",
                    "status": "success",
                    "records_count": result.get("written", 0),
                    "duration_sec": round(duration_sec, 2)
                }
                emit_structured_log(structured_log)
        else:
            ACTIVE_TASKS[task_id].update({
                "status": "failed",
                "progress": "N/A",
                "message": result.get("message", "Échec de l'ingestion"),
                "completed_at": datetime.now(),
                "error": result.get("error")
            })
            task_logger.error(f"[{task_id}] ❌ Échec de l'ingestion: {result.get('error')}")
            
            # Log structuré si Grafana activé - via ROOT logger
            if ENABLE_GRAFANA_LOGS:
                structured_log = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "ingestion_failed",
                    "task_id": task_id,
                    "task_type": "france_travail_offers",
                    "status": "failed",
                    "error": result.get("error"),
                    "duration_sec": round(duration_sec, 2)
                }
                emit_structured_log(structured_log)
    except Exception as e:
        duration_sec = time.time() - start_time
        ACTIVE_TASKS[task_id].update({
            "status": "failed",
            "progress": "N/A",
            "message": f"Erreur: {str(e)}",
            "completed_at": datetime.now(),
            "error": str(e)
        })
        task_logger.error(
            f"[{task_id}] ❌ Exception après {duration_sec:.2f}s: {e}", 
            exc_info=True
        )
        
        # Log structuré si Grafana activé - via ROOT logger
        if ENABLE_GRAFANA_LOGS:
            structured_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "ingestion_error",
                "task_id": task_id,
                "task_type": "france_travail_offers",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_sec": round(duration_sec, 2)
            }
            emit_structured_log(structured_log)


def run_welcome_to_jungle_task(task_id: str, mode: str, max_jobs: int, max_companies: int):
    """Wrapper pour l'ingestion WTTJ avec mise à jour du statut"""
    
    # Logger spécifique pour cette ingestion
    task_logger = get_endpoint_logger('wttj', 'ingestion')
    
    def update_progress(segment: str, current: int, total: int, ok: int, ko: int):
        """Callback pour mettre à jour la progression en temps réel"""
        progress_pct = int((current / total) * 100) if total > 0 else 0
        ACTIVE_TASKS[task_id].update({
            "progress": f"{progress_pct}%",
            "message": f"Segment {segment}: {current}/{total} URLs traitées (✓{ok} ✗{ko})",
            "current_segment": segment,
            "current_url": current,
            "total_urls": total
        })
    
    try:
        task_logger.info(f"[{task_id}] 📥 Début de l'ingestion Welcome to the Jungle (mode={mode})")
        result = ingest_welcome_to_the_jungle(
            storage=None,
            mode=mode,
            max_jobs=max_jobs,
            max_companies=max_companies,
            progress_callback=update_progress
        )
        
        if result["success"]:
            ACTIVE_TASKS[task_id].update({
                "status": "completed",
                "progress": "100%",
                "message": result["message"],
                "completed_at": datetime.now(),
                "result": {
                    "run_id": result.get("run_id"),
                    "dt": result.get("dt"),
                    "mode": result.get("mode"),
                    "total_processed": result.get("total_processed"),
                    "total_written": result.get("total_written"),
                    "elapsed_s": result.get("elapsed_s"),
                    "jobs": result.get("jobs"),
                    "companies": result.get("companies")
                }
            })
            task_logger.info(f"[{task_id}] ✅ Ingestion terminée: {result.get('total_written')} records")
        else:
            ACTIVE_TASKS[task_id].update({
                "status": "failed",
                "progress": "N/A",
                "message": result.get("message", "Échec de l'ingestion"),
                "completed_at": datetime.now(),
                "error": result.get("error")
            })
            task_logger.error(f"[{task_id}] ❌ Échec: {result.get('error')}")
    except Exception as e:
        ACTIVE_TASKS[task_id].update({
            "status": "failed",
            "progress": "N/A",
            "message": f"Erreur: {str(e)}",
            "completed_at": datetime.now(),
            "error": str(e)
        })
        task_logger.error(f"[{task_id}] ❌ Exception: {e}", exc_info=True)


def run_merge_datasets_task(task_id: str, ft_prefix: Optional[str], wttj_prefix: Optional[str], 
                             output_prefix: Optional[str], output_format: str):
    """Wrapper pour la fusion des datasets avec mise à jour du statut"""
    
    # Logger spécifique pour cette opération
    task_logger = get_endpoint_logger('merge_datasets', 'ingestion')
    
    def update_progress(step: str, message: str):
        """Callback pour mettre à jour la progression en temps réel"""
        ACTIVE_TASKS[task_id].update({
            "progress": step,
            "message": message,
            "current_step": step
        })
    
    try:
        task_logger.info(f"[{task_id}] 🔀 Début de la fusion FT + WTTJ")
        result = merge_ft_wttj_datasets(
            ft_prefix=ft_prefix,
            wttj_prefix=wttj_prefix,
            output_prefix=output_prefix,
            output_format=output_format,
            progress_callback=update_progress
        )
        
        if result["success"]:
            ACTIVE_TASKS[task_id].update({
                "status": "completed",
                "progress": "100%",
                "message": result["message"],
                "completed_at": datetime.now(),
                "result": {
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
            })
            task_logger.info(f"[{task_id}] ✅ Fusion terminée: {result.get('total_offers')} offres")
        else:
            ACTIVE_TASKS[task_id].update({
                "status": "failed",
                "progress": "N/A",
                "message": result.get("message", "Échec de la fusion"),
                "completed_at": datetime.now(),
                "error": result.get("error")
            })
            task_logger.error(f"[{task_id}] ❌ Échec: {result.get('error')}")
    except Exception as e:
        ACTIVE_TASKS[task_id].update({
            "status": "failed",
            "progress": "N/A",
            "message": f"Erreur: {str(e)}",
            "completed_at": datetime.now(),
            "error": str(e)
        })
        task_logger.error(f"[{task_id}] ❌ Exception: {e}", exc_info=True)


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

global rome_model
rome_model = get_rome_model()


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
    # Logger spécifique pour les prédictions
    prediction_logger = get_endpoint_logger('rome_prediction', 'prediction')
    
    prediction_logger.info(f"Requête de prédiction - Intitulé: {req.intitule[:50] if req.intitule else 'N/A'}")
    
    text = build_text_payload(
        intitule=req.intitule,
        description=req.description,
        competences=req.competences,
    )

    pred = predict_top_k(ARTIFACTS, text, top_k=TOP_K, rome_index=rome_model)
    
    prediction_logger.info(f"Prédiction réussie - Code ROME: {pred.get('code_rome', 'N/A')}")
    
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
        task_id = f"rome-metiers-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        
        # Enregistrer la tâche
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_rome_metiers",
            "status": "running",
            "started_at": datetime.now(),
            "progress": "0%",
            "message": "Ingestion des codes ROME métiers en cours..."
        }
        
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
            # Utiliser le logger spécifique
            task_logger = get_endpoint_logger('rome_metiers', 'ingestion')
            
            task_logger.info("📥 Début de l'ingestion synchrone des codes ROME")
            
            result = ingest_rome_metiers()
            
            if result["success"]:
                task_logger.info(f"✅ Ingestion réussie: {result['records_count']} codes ROME")
            else:
                task_logger.error(f"❌ Échec de l'ingestion: {result.get('error')}")
                
            return IngestResponse(**result)
            
        except Exception as e:
            task_logger.error(f"❌ Erreur lors de l'ingestion: {e}", exc_info=True)
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
    current_time = datetime.now()
    tasks_to_remove = []
    for task_id, task_info in ACTIVE_TASKS.items():
        if task_info.get("status") == "completed":
            completed_at = task_info.get("completed_at")
            if completed_at and (current_time - completed_at).total_seconds() > 300:
                tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        del ACTIVE_TASKS[task_id]
    
    # Filtrer les tâches d'ingestion
    ingestion_operations = [
        "ingest_rome_metiers",
        "ingest_france_travail_offers",
        "ingest_welcome_to_jungle"
    ]
    
    return {
        "status": "ok",
        "active_tasks": [
            {
                "task_id": task_id,
                "operation": task_info.get("operation"),
                "status": task_info.get("status"),
                "started_at": task_info.get("started_at").isoformat() if task_info.get("started_at") else None,
                "progress": task_info.get("progress"),
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
    current_time = datetime.now()
    tasks_to_remove = []
    for task_id, task_info in ACTIVE_TASKS.items():
        if task_info.get("status") == "completed":
            completed_at = task_info.get("completed_at")
            if completed_at and (current_time - completed_at).total_seconds() > 300:
                tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        del ACTIVE_TASKS[task_id]
    
    # Filtrer les tâches de data processing
    data_operations = [
        "merge_datasets"
    ]
    
    return {
        "status": "ok",
        "active_tasks": [
            {
                "task_id": task_id,
                "operation": task_info.get("operation"),
                "status": task_info.get("status"),
                "started_at": task_info.get("started_at").isoformat() if task_info.get("started_at") else None,
                "progress": task_info.get("progress"),
                "message": task_info.get("message")
            }
            for task_id, task_info in ACTIVE_TASKS.items()
            if task_info.get("operation") in data_operations
        ],
        "available_operations": [
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
        "progress": task_info.get("progress"),
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
        task_id = f"ft-offers-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        
        # Enregistrer la tâche
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_france_travail_offers",
            "status": "running",
            "started_at": datetime.now(),
            "progress": "0%",
            "message": f"Ingestion des offres France Travail en cours (max_rome_codes: {max_rome_codes or 'tous'})...",
            "params": {
                "max_rome_codes": max_rome_codes,
                "window_days": window_days,
                "max_windows": max_windows
            }
        }
        
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
            # Utiliser le logger spécifique
            task_logger = get_endpoint_logger('france_travail_offers', 'ingestion')
            
            task_logger.info("📥 Début de l'ingestion synchrone des offres France Travail")
            result = ingest_france_travail_offers(
                storage=None,
                client=None,
                window_days=window_days,
                max_windows=max_windows,
                binary_split_min_seconds=binary_split_min_seconds,
                max_rome_codes=max_rome_codes,
                logger_override=task_logger
            )
            
            if result["success"]:
                task_logger.info(f"✅ Ingestion réussie: {result['written']} offres, {result['rome_processed']} codes ROME")
            else:
                task_logger.error(f"❌ Échec de l'ingestion: {result.get('error')}")
                
            return IngestOffersResponse(**result)
            
        except Exception as e:
            task_logger.error(f"❌ Erreur lors de l'ingestion des offres: {e}", exc_info=True)
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
    max_companies: int = Query(0, description="Limiter le nombre de companies (0 = tous)")
):
    """Ingestion des offres d'emploi Welcome to the Jungle en couche bronze.

Collecte les URLs depuis les sitemaps et extrait les données structurées
des pages jobs et companies pour stockage en bronze.
"""
    logger.info(f"Requête d'ingestion WTTJ reçue (background={background}, mode={mode}, max_jobs={max_jobs}, max_companies={max_companies})")
    
    if background:
        # Générer un task_id unique
        task_id = f"wttj-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        
        # Enregistrer la tâche
        ACTIVE_TASKS[task_id] = {
            "operation": "ingest_welcome_to_jungle",
            "status": "running",
            "started_at": datetime.now(),
            "progress": "0%",
            "message": f"Ingestion WTTJ en cours (mode: {mode}, jobs: {max_jobs or 'tous'}, companies: {max_companies or 'tous'})...",
            "params": {
                "mode": mode,
                "max_jobs": max_jobs,
                "max_companies": max_companies
            }
        }
        
        # Lancer en arrière-plan avec wrapper
        logger.info("Lancement de l'ingestion WTTJ en arrière-plan")
        background_tasks.add_task(
            run_welcome_to_jungle_task,
            task_id,
            mode,
            max_jobs,
            max_companies
        )
        
        return IngestWTTJResponse(
            success=True,
            message=f"Ingestion WTTJ lancée en arrière-plan (task_id: {task_id})",
            run_id=task_id
        )
    else:
        # Exécution synchrone
        try:
            # Utiliser le logger spécifique
            task_logger = get_endpoint_logger('wttj', 'ingestion')
            
            task_logger.info("📥 Début de l'ingestion synchrone WTTJ")
            result = ingest_welcome_to_the_jungle(
                storage=None,
                mode=mode,
                max_jobs=max_jobs,
                max_companies=max_companies
            )
            
            if result["success"]:
                task_logger.info(f"✅ Ingestion réussie: {result.get('total_written')} records")
            else:
                task_logger.error(f"❌ Échec de l'ingestion: {result.get('error')}")
                
            return IngestWTTJResponse(**result)
            
        except Exception as e:
            task_logger.error(f"❌ Erreur lors de l'ingestion WTTJ: {e}", exc_info=True)
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

Cette opération lit les données des couches Bronze (FT) et Silver (WTTJ), les normalise,
les fusionne et les déduplique pour créer un dataset d'entraînement unifié.

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

# Fusion avec préfixes spécifiques
curl -X POST "http://localhost:8000/data/merge-datasets?ft_prefix=bronze/offers&wttj_prefix=silver/jobs"

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

Lit les données des couches Bronze (FT) et Silver (WTTJ), normalise selon
le modèle Silver_Datamodel, fusionne et déduplique pour créer un dataset unifié.

**Étapes:**

1. Détection automatique des préfixes si non spécifiés
2. Lecture et normalisation des données FT Bronze
3. Lecture et normalisation des données WTTJ Silver
4. Fusion et déduplication par URL
5. Calcul des statistiques
6. Sauvegarde du dataset fusionné
"""
    logger.info(f"Requête de fusion datasets reçue (background={background}, format={output_format})")
    
    if background:
        # Générer un task_id unique
        task_id = f"merge-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        
        # Enregistrer la tâche
        ACTIVE_TASKS[task_id] = {
            "operation": "merge_datasets",
            "status": "running",
            "started_at": datetime.now(),
            "progress": "0%",
            "message": f"Fusion des datasets en cours (format: {output_format})...",
            "params": {
                "ft_prefix": ft_prefix,
                "wttj_prefix": wttj_prefix,
                "output_prefix": output_prefix,
                "output_format": output_format
            }
        }
        
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
            # Utiliser le logger spécifique
            task_logger = get_endpoint_logger('merge_datasets', 'ingestion')
            
            task_logger.info("🔀 Début de la fusion synchrone des datasets")
            result = merge_ft_wttj_datasets(
                ft_prefix=ft_prefix,
                wttj_prefix=wttj_prefix,
                output_prefix=output_prefix,
                output_format=output_format
            )
            
            if result["success"]:
                task_logger.info(f"✅ Fusion réussie: {result.get('total_offers')} offres fusionnées")
            else:
                task_logger.error(f"❌ Échec de la fusion: {result.get('error')}")
                
            return MergeDatasetResponse(**result)
            
        except Exception as e:
            task_logger.error(f"❌ Erreur lors de la fusion: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de la fusion: {str(e)}"
            )
