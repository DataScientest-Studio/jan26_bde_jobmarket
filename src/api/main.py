import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

# IMPORTANT: import depuis ton module de prédiction
# ajuste le chemin selon ton projet (ex: from src.models.predict_model import ...)
from src.models.predict_model import build_text_payload, load_artifacts, predict_top_k, get_rome_model
from src.ingest.bronze.france_travail_rome_metiers import ingest_rome_metiers
from src.ingest.bronze.france_travail import ingest_france_travail_offers

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


class PredictRequest(BaseModel):
    """Requête de prédiction de code ROME pour une offre d'emploi"""
    intitule: Optional[str] = Field(None, description="Titre du poste", example="Développeur Python Senior")
    description: Optional[str] = Field(None, description="Description détaillée du poste", example="Développement d'applications web avec Python, FastAPI, PostgreSQL")
    competences: Optional[List[str]] = Field(None, description="Liste des compétences techniques", example=["Python", "FastAPI", "SQL", "Docker"])
    
    class Config:
        schema_extra = {
            "example": {
                "intitule": "Data Scientist Senior",
                "description": "Analyse de données, machine learning, déploiement de modèles en production",
                "competences": ["Python", "Scikit-learn", "TensorFlow", "SQL"]
            }
        }


class PredictResponse(BaseModel):
    """Résultat de la prédiction avec les codes ROME les plus probables"""
    model_name: Optional[str] = Field(None, description="Nom du modèle utilisé")
    model_version: Optional[str] = Field(None, description="Version du modèle")
    rome_pred: str = Field(..., description="Code ROME prédit (le plus probable)", example="M1805")
    rome_label: Optional[str] = Field(None, description="Libellé du code ROME prédit", example="Études et développement informatique")
    top_k: List[dict] = Field(..., description="Top K prédictions avec scores", example=[
        {"rome_code": "M1805", "score": 0.89, "label": "Études et développement informatique"},
        {"rome_code": "M1806", "score": 0.76, "label": "Conseil et maîtrise d'ouvrage en systèmes d'information"}
    ])


class IngestResponse(BaseModel):
    """Résultat d'une opération d'ingestion de données"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    key: Optional[str] = Field(None, description="Clé de stockage des données", example="bronze/rome/rome_metiers.jsonl")
    records_count: Optional[int] = Field(None, description="Nombre total de codes ROME", example=532)
    records_written: Optional[int] = Field(None, description="Nombre d'enregistrements écrits", example=532)
    error: Optional[str] = Field(None, description="Message d'erreur si échec")


class IngestOffersResponse(BaseModel):
    """Résultat d'une opération d'ingestion des offres d'emploi"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    run_id: Optional[str] = Field(None, description="Identifiant unique du run", example="20260223T120000Z")
    run_key: Optional[str] = Field(None, description="Clé des métadonnées du run")
    rome_processed: Optional[int] = Field(None, description="Nombre de codes ROME traités", example=532)
    calls: Optional[int] = Field(None, description="Nombre d'appels API effectués", example=1500)
    written: Optional[int] = Field(None, description="Nombre total d'offres écrites", example=15000)
    elapsed_s: Optional[float] = Field(None, description="Durée de l'ingestion en secondes", example=3600.5)
    errors: Optional[int] = Field(None, description="Nombre d'erreurs rencontrées", example=0)
    error: Optional[str] = Field(None, description="Message d'erreur si échec")

@app.on_event("startup")
def _startup_load_model():
    """
    Charge les artefacts une seule fois au démarrage.
    Evite de recharger MinIO/joblib à chaque requête.
    """
    global ARTIFACTS
    ARTIFACTS = load_artifacts()
    # Optionnel: log
    print(f"✅ API loaded model: {MODEL_NAME} / {ARTIFACTS['version']}")


# =====================================
# Wrappers pour tâches d'ingestion avec tracking
# =====================================

def run_rome_metiers_task(task_id: str):
    """Wrapper pour l'ingestion des codes ROME avec mise à jour du statut"""
    try:
        logger.info(f"[{task_id}] Début de l'ingestion des codes ROME")
        result = ingest_rome_metiers()
        
        if result["success"]:
            ACTIVE_TASKS[task_id].update({
                "status": "completed",
                "progress": "100%",
                "message": result["message"],
                "completed_at": datetime.now(),
                "result": {
                    "records_count": result.get("records_count"),
                    "records_written": result.get("records_written"),
                    "key": result.get("key")
                }
            })
            logger.info(f"[{task_id}] Ingestion terminée avec succès")
        else:
            ACTIVE_TASKS[task_id].update({
                "status": "failed",
                "progress": "N/A",
                "message": result.get("message", "Échec de l'ingestion"),
                "completed_at": datetime.now(),
                "error": result.get("error")
            })
            logger.error(f"[{task_id}] Échec de l'ingestion: {result.get('error')}")
    except Exception as e:
        ACTIVE_TASKS[task_id].update({
            "status": "failed",
            "progress": "N/A",
            "message": f"Erreur: {str(e)}",
            "completed_at": datetime.now(),
            "error": str(e)
        })
        logger.error(f"[{task_id}] Exception: {e}", exc_info=True)


def run_france_travail_offers_task(task_id: str, window_days: int, max_windows: int, 
                                     binary_split_min_seconds: int, max_rome_codes: int):
    """Wrapper pour l'ingestion des offres FT avec mise à jour du statut"""
    
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
        logger.info(f"[{task_id}] Début de l'ingestion des offres France Travail")
        result = ingest_france_travail_offers(
            storage=None,
            client=None,
            window_days=window_days,
            max_windows=max_windows,
            binary_split_min_seconds=binary_split_min_seconds,
            max_rome_codes=max_rome_codes,
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
                    "run_key": result.get("run_key"),
                    "rome_processed": result.get("rome_processed"),
                    "calls": result.get("calls"),
                    "written": result.get("written"),
                    "elapsed_s": result.get("elapsed_s"),
                    "errors": result.get("errors")
                }
            })
            logger.info(f"[{task_id}] Ingestion terminée: {result.get('written')} offres")
        else:
            ACTIVE_TASKS[task_id].update({
                "status": "failed",
                "progress": "N/A",
                "message": result.get("message", "Échec de l'ingestion"),
                "completed_at": datetime.now(),
                "error": result.get("error")
            })
            logger.error(f"[{task_id}] Échec: {result.get('error')}")
    except Exception as e:
        ACTIVE_TASKS[task_id].update({
            "status": "failed",
            "progress": "N/A",
            "message": f"Erreur: {str(e)}",
            "completed_at": datetime.now(),
            "error": str(e)
        })
        logger.error(f"[{task_id}] Exception: {e}", exc_info=True)


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
    text = build_text_payload(
        intitule=req.intitule,
        description=req.description,
        competences=req.competences,
    )

    pred = predict_top_k(ARTIFACTS, text, top_k=TOP_K, rome_index=rome_model)
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
            logger.info("Début de l'ingestion synchrone")
            result = ingest_rome_metiers()
            
            if result["success"]:
                logger.info(f"Ingestion réussie: {result['records_count']} codes ROME")
            else:
                logger.error(f"Échec de l'ingestion: {result.get('error')}")
                
            return IngestResponse(**result)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ingestion: {e}", exc_info=True)
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
            }
        ]
    }


@app.get(
    "/ingest/tasks/{task_id}",
    tags=["Ingestion"],
    summary="Détails d'une tâche d'ingestion",
    description="Récupère les informations détaillées d'une tâche d'ingestion spécifique"
)
def get_task_details(task_id: str):
    """Retourne les détails complets d'une tâche d'ingestion.

**Utilisation:**

```bash
# Après avoir lancé une ingestion en arrière-plan, récupérer son statut
curl http://localhost:8000/ingest/tasks/ft-offers-20260223T214500
```
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
            logger.info("Début de l'ingestion synchrone des offres")
            result = ingest_france_travail_offers(
                storage=None,
                client=None,
                window_days=window_days,
                max_windows=max_windows,
                binary_split_min_seconds=binary_split_min_seconds,
                max_rome_codes=max_rome_codes
            )
            
            if result["success"]:
                logger.info(f"Ingestion réussie: {result['written']} offres, {result['rome_processed']} codes ROME")
            else:
                logger.error(f"Échec de l'ingestion: {result.get('error')}")
                
            return IngestOffersResponse(**result)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ingestion des offres: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de l'ingestion: {str(e)}"
            )
